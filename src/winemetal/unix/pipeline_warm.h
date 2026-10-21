/* Local, best-effort pipeline warm-up. Included after the normal PSO builders
 * so it uses exactly their descriptor conversion, not a second approximation.
 * No PSOs are kept: the benefit is Metal's compiled-code cache. */
#define MADEIRA_WARM_PENDING 128
#define MADEIRA_WARM_ARCHIVES 8
struct madeira_warm_request { uint64_t device, info, vd; };
struct madeira_warm_job {
  obj_handle_t device;
  unsigned kind, narchives;
  struct WMTRenderPipelineInfo render;
  struct WMTComputePipelineInfo compute;
  struct WMTVertexDescriptorInfo vd;
  obj_handle_t archives[MADEIRA_WARM_ARCHIVES];
};
static dispatch_queue_t madeira_warm_queues[2];
static unsigned madeira_warm_workers;
static _Atomic unsigned madeira_warm_pending, madeira_warm_sent, madeira_warm_done;
static _Atomic unsigned madeira_warm_skipped, madeira_warm_failed;
static _Atomic uint64_t madeira_warm_us;

static int madeira_warm_headroom(void) {
  task_vm_info_data_t vmi;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vmi, &count) != KERN_SUCCESS) return 0;
  return madeira_avail_memory(vmi.phys_footprint) >= (256ull << 20);
}
static void madeira_warm_release(struct madeira_warm_job *job) {
  [(id)(uintptr_t)job->device release];
  if (job->kind == 2) [(id)(uintptr_t)job->compute.compute_function release];
  else {
    [(id)(uintptr_t)job->render.vertex_function release];
    [(id)(uintptr_t)job->render.fragment_function release];
  }
  for (unsigned i = 0; i < job->narchives; i++) [(id)(uintptr_t)job->archives[i] release];
  free(job);
  atomic_fetch_sub(&madeira_warm_pending, 1);
}
static void madeira_warm_execute(struct madeira_warm_job *job) {
  @autoreleasepool {
    obj_handle_t pso = 0;
    double begin = CACurrentMediaTime();
    if (madeira_warm_headroom()) {
      if (job->kind == 2) {
        struct unixcall_mtldevice_newcomputepso a = {0};
        a.device = job->device; a.info.ptr = &job->compute;
        _MTLDevice_newComputePipelineState(&a); pso = a.ret_pso;
      } else if (job->kind == 1) {
        struct unixcall_mtldevice_newrenderpso_vd a = {0};
        a.device = job->device; a.info.ptr = &job->render; a.vd.ptr = &job->vd;
        _MTLDevice_newRenderPipelineStateVD(&a); pso = a.ret_pso;
      } else {
        struct unixcall_mtldevice_newrenderpso a = {0};
        a.device = job->device; a.info.ptr = &job->render;
        _MTLDevice_newRenderPipelineState(&a); pso = a.ret_pso;
      }
      if (!pso) atomic_fetch_add(&madeira_warm_failed, 1);
      [(id)(uintptr_t)pso release];
    } else atomic_fetch_add(&madeira_warm_skipped, 1);
    atomic_fetch_add(&madeira_warm_us, (uint64_t)((CACurrentMediaTime() - begin) * 1000000));
    unsigned done = atomic_fetch_add(&madeira_warm_done, 1) + 1;
    madeira_warm_release(job);
    if (done == 1 || !(done % 128))
      fprintf(stderr, "[pso-warm] completed=%u submitted=%u skipped=%u failed=%u pending=%u work=%llu ms\n",
              done, atomic_load(&madeira_warm_sent), atomic_load(&madeira_warm_skipped),
              atomic_load(&madeira_warm_failed), atomic_load(&madeira_warm_pending),
              (unsigned long long)(atomic_load(&madeira_warm_us) / 1000));
  }
}
static int madeira_warm_submit(const struct madeira_warm_request *r, unsigned kind) {
  static dispatch_once_t once;
  if (wmtr_enabled() || !r || !r->device || !r->info || kind > 2 || (kind == 1 && !r->vd)) return 0;
  dispatch_once(&once, ^{
    long long n = madeira_cfg_int("pso-warm", 0);
    madeira_warm_workers = n <= 0 ? 0 : n > 2 ? 2 : (unsigned)n;
    for (unsigned i = 0; i < madeira_warm_workers; i++)
      madeira_warm_queues[i] = dispatch_queue_create("madeira.pso-warm",
          dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
    if (madeira_warm_workers)
      fprintf(stderr, "[pso-warm] native handler active: %u workers, %u pending cap, 256MB headroom floor\n",
              madeira_warm_workers, MADEIRA_WARM_PENDING);
  });
  if (!madeira_warm_workers) return 0;
  if (!madeira_warm_headroom()) { atomic_fetch_add(&madeira_warm_skipped, 1); return 1; }
  if (atomic_fetch_add(&madeira_warm_pending, 1) >= MADEIRA_WARM_PENDING) {
    atomic_fetch_sub(&madeira_warm_pending, 1);
    atomic_fetch_add(&madeira_warm_skipped, 1); return 1;
  }
  struct madeira_warm_job *job = calloc(1, sizeof(*job));
  if (!job) {
    atomic_fetch_sub(&madeira_warm_pending, 1);
    atomic_fetch_add(&madeira_warm_skipped, 1); return 1;
  }
  job->kind = kind; job->device = r->device;
  struct WMTConstMemoryPointer *lookup;
  if (kind == 2) {
    memcpy(&job->compute, (const void *)(uintptr_t)r->info, sizeof(job->compute));
    job->narchives = job->compute.num_binary_archives_for_lookup;
    lookup = &job->compute.binary_archives_for_lookup;
    job->compute.binary_archive_for_serialization = 0; /* never mutate caller-owned archives */
  } else {
    memcpy(&job->render, (const void *)(uintptr_t)r->info, sizeof(job->render));
    if (kind == 1) memcpy(&job->vd, (const void *)(uintptr_t)r->vd, sizeof(job->vd));
    job->narchives = job->render.num_binary_archives_for_lookup;
    lookup = &job->render.binary_archives_for_lookup;
    job->render.binary_archive_for_serialization = 0;
  }
  if (job->narchives > MADEIRA_WARM_ARCHIVES || (job->narchives && !lookup->ptr)) {
    free(job); atomic_fetch_sub(&madeira_warm_pending, 1);
    atomic_fetch_add(&madeira_warm_skipped, 1); return 1;
  }
  if (job->narchives) memcpy(job->archives, lookup->ptr, job->narchives * sizeof(obj_handle_t));
  lookup->ptr = job->archives; /* deep copy every caller-owned pointer before returning */
  [(id)(uintptr_t)job->device retain];
  if (kind == 2) [(id)(uintptr_t)job->compute.compute_function retain];
  else {
    [(id)(uintptr_t)job->render.vertex_function retain];
    [(id)(uintptr_t)job->render.fragment_function retain];
  }
  for (unsigned i = 0; i < job->narchives; i++) [(id)(uintptr_t)job->archives[i] retain];
  unsigned slot = atomic_fetch_add(&madeira_warm_sent, 1) % madeira_warm_workers;
  dispatch_async(madeira_warm_queues[slot], ^{ madeira_warm_execute(job); });
  return 1; /* full/budget-skipped is supported too: don't disable the PE caller */
}
