// The training interface, present so the engine links and honest about what
// it does: nothing. gk has no gradients, so every entry point that would
// start training aborts with a clear message instead of computing garbage.
// Inference never reaches any of these.

#include "ggml-compat-impl.h"

#include "ggml-opt.h"

#define GGML_OPT_UNSUPPORTED() \
    GGML_ABORT("training (ggml_opt) is not supported by the gk engine")

ggml_opt_dataset_t ggml_opt_dataset_init(enum ggml_type type_data, enum ggml_type type_label,
                                         int64_t ne_datapoint, int64_t ne_label,
                                         int64_t ndata, int64_t ndata_shard) {
    GGML_UNUSED(type_data); GGML_UNUSED(type_label); GGML_UNUSED(ne_datapoint);
    GGML_UNUSED(ne_label); GGML_UNUSED(ndata); GGML_UNUSED(ndata_shard);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    GGML_UNUSED(dataset);
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    GGML_UNUSED(dataset);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    GGML_UNUSED(dataset);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    GGML_UNUSED(dataset);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(dataset); GGML_UNUSED(idata);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch,
                                struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_UNUSED(dataset); GGML_UNUSED(data_batch); GGML_UNUSED(labels_batch); GGML_UNUSED(ibatch);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch,
                                     void * labels_batch, int64_t ibatch) {
    GGML_UNUSED(dataset); GGML_UNUSED(data_batch); GGML_UNUSED(nb_data_batch);
    GGML_UNUSED(labels_batch); GGML_UNUSED(ibatch);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);
    struct ggml_opt_optimizer_params p;
    memset(&p, 0, sizeof(p));
    return p;
}

struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);
    struct ggml_opt_optimizer_params p;
    memset(&p, 0, sizeof(p));
    return p;
}

struct ggml_opt_params ggml_opt_default_params(ggml_backend_sched_t backend_sched,
                                               enum ggml_opt_loss_type loss_type) {
    struct ggml_opt_params p;
    memset(&p, 0, sizeof(p));
    p.backend_sched = backend_sched;
    p.loss_type     = loss_type;
    return p;
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    GGML_UNUSED(params);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx); // freeing a context that could never exist is a no-op
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(optimizer);
    GGML_OPT_UNSUPPORTED();
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(node);
    GGML_OPT_UNSUPPORTED();
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
    GGML_OPT_UNSUPPORTED();
}

const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type type) {
    GGML_UNUSED(type);
    return "unsupported";
}

ggml_opt_result_t ggml_opt_result_init(void) {
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    GGML_UNUSED(result);
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    GGML_UNUSED(result);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    GGML_UNUSED(result); GGML_UNUSED(ndata);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    GGML_UNUSED(result); GGML_UNUSED(loss); GGML_UNUSED(unc);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    GGML_UNUSED(result); GGML_UNUSED(pred);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    GGML_UNUSED(result); GGML_UNUSED(accuracy); GGML_UNUSED(unc);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_prepare_alloc(ggml_opt_context_t opt_ctx, struct ggml_context * ctx_compute,
                            struct ggml_cgraph * gf, struct ggml_tensor * inputs,
                            struct ggml_tensor * outputs) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(ctx_compute); GGML_UNUSED(gf);
    GGML_UNUSED(inputs); GGML_UNUSED(outputs);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(backward);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(result);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_epoch(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset,
                    ggml_opt_result_t result_train, ggml_opt_result_t result_eval,
                    int64_t idata_split, ggml_opt_epoch_callback callback_train,
                    ggml_opt_epoch_callback callback_eval) {
    GGML_UNUSED(opt_ctx); GGML_UNUSED(dataset); GGML_UNUSED(result_train);
    GGML_UNUSED(result_eval); GGML_UNUSED(idata_split);
    GGML_UNUSED(callback_train); GGML_UNUSED(callback_eval);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_epoch_callback_progress_bar(bool train, ggml_opt_context_t opt_ctx,
                    ggml_opt_dataset_t dataset, ggml_opt_result_t result,
                    int64_t ibatch, int64_t ibatch_max, int64_t t_start_us) {
    GGML_UNUSED(train); GGML_UNUSED(opt_ctx); GGML_UNUSED(dataset); GGML_UNUSED(result);
    GGML_UNUSED(ibatch); GGML_UNUSED(ibatch_max); GGML_UNUSED(t_start_us);
    GGML_OPT_UNSUPPORTED();
}

void ggml_opt_fit(ggml_backend_sched_t backend_sched, struct ggml_context * ctx_compute,
                  struct ggml_tensor * inputs, struct ggml_tensor * outputs,
                  ggml_opt_dataset_t dataset, enum ggml_opt_loss_type loss_type,
                  enum ggml_opt_optimizer_type optimizer,
                  ggml_opt_get_optimizer_params get_opt_pars, int64_t nepoch,
                  int64_t nbatch_logical, float val_split, bool silent) {
    GGML_UNUSED(backend_sched); GGML_UNUSED(ctx_compute); GGML_UNUSED(inputs);
    GGML_UNUSED(outputs); GGML_UNUSED(dataset); GGML_UNUSED(loss_type);
    GGML_UNUSED(optimizer); GGML_UNUSED(get_opt_pars); GGML_UNUSED(nepoch);
    GGML_UNUSED(nbatch_logical); GGML_UNUSED(val_split); GGML_UNUSED(silent);
    GGML_OPT_UNSUPPORTED();
}
