#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec256/vec256.h>

// mainly adapted from https://github.com/intel/intel-extension-for-pytorch/blob/master/csrc/cpu/aten/kernels/optimizer/AdamFusedStepKrnl.cpp

int ds_adam_step_bf16_states(int optimizer_id,
                             size_t step,
                             float lr,
                             float beta1,
                             float beta2,
                             float eps,
                             float weight_decay,
                             bool bias_correction,
                             torch::Tensor& params,
                             torch::Tensor& grads,
                             torch::Tensor& exp_avg,
                             torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    assert(params.scalar_type() == at::kFloat);
    assert(exp_avg.scalar_type() == at::kBFloat16);
    assert(exp_avg_sq.scalar_type() == at::kBFloat16);
    assert(grads.scalar_type() == at::kFloat);

    float* param_data = params_c.data_ptr<float>();
    at::BFloat16* exp_avg_data = exp_avg_c.data_ptr<at::BFloat16>();
    at::BFloat16* exp_avg_sq_data = exp_avg_sq_c.data_ptr<at::BFloat16>();
    float* grad_data = grads_c.data_ptr<float>();

    float bias_correction1 = 1 - std::pow(beta1, step);
    float step_size = lr / bias_correction1;
    float bias_correction2 = 1 - std::pow(beta2, step);

    float exp_avg_grad_coefficient = float(1 - beta1);
    float exp_avg_sq_grad_coefficient = float(1 - beta2);

    using bVec = at::vec::Vectorized<at::BFloat16>;
    using fVec = at::vec::Vectorized<float>;

    int64_t grain_size = 512;

    at::parallel_for(
      0, params_c.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        float* param_ptr = param_data + begin;
        at::BFloat16* exp_avg_ptr = exp_avg_data + begin;
        at::BFloat16* exp_avg_sq_ptr = exp_avg_sq_data + begin;
        float* grad_ptr = grad_data + begin;

        const int64_t size = end - begin;

        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          // load grad vec
          fVec grad_fvec = fVec::loadu(grad_ptr + d);
          fVec grad_fvec2 = fVec::loadu(grad_ptr + d + fVec::size());
          // load param vec
          fVec param_fvec = fVec::loadu(param_ptr + d);
          fVec param_fvec2 = fVec::loadu(param_ptr + d + fVec::size());
          // skip weight decay for now
          // update exp_avg, exp_avg_sq
          fVec exp_avg_fvec_tmp, exp_avg_fvec2_tmp;
          std::tie(exp_avg_fvec_tmp, exp_avg_fvec2_tmp) = convert_bfloat16_float(bVec::loadu(exp_avg_ptr + d));
          fVec exp_avg_fvec = exp_avg_fvec_tmp * fVec(beta1) + grad_fvec * fVec(exp_avg_grad_coefficient);
          fVec exp_avg_fvec2 =  exp_avg_fvec2_tmp * fVec(beta1) + grad_fvec2 * fVec(exp_avg_grad_coefficient);
          bVec exp_avg_bvec = convert_float_bfloat16(exp_avg_fvec, exp_avg_fvec2);
          exp_avg_bvec.store(exp_avg_ptr + d);

          fVec exp_avg_sq_fvec_tmp, exp_avg_sq_fvec2_tmp;
          std::tie(exp_avg_sq_fvec_tmp, exp_avg_sq_fvec2_tmp) = convert_bfloat16_float(bVec::loadu(exp_avg_sq_ptr + d));
          fVec exp_avg_sq_fvec = exp_avg_sq_fvec_tmp * fVec(beta2) + grad_fvec * grad_fvec * fVec(exp_avg_sq_grad_coefficient);
          fVec exp_avg_sq_fvec2 = exp_avg_sq_fvec2_tmp * fVec(beta2) + grad_fvec2 * grad_fvec2 * fVec(exp_avg_sq_grad_coefficient);
          bVec exp_avg_sq_bvec = convert_float_bfloat16(exp_avg_sq_fvec, exp_avg_sq_fvec2);
          exp_avg_sq_bvec.store(exp_avg_sq_ptr + d);

          fVec denom_fvec, denom_fvec2;
          denom_fvec =
                (exp_avg_sq_fvec / fVec(bias_correction2)).sqrt() + fVec(eps);
          denom_fvec2 =
                (exp_avg_sq_fvec2 / fVec(bias_correction2)).sqrt() + fVec(eps);

          // update param
          param_fvec = param_fvec - fVec(step_size) * exp_avg_fvec / denom_fvec;
          param_fvec2 =
              param_fvec2 - fVec(step_size) * exp_avg_fvec2 / denom_fvec2;
          param_fvec.store(param_ptr + d);
          param_fvec2.store(param_ptr + d + fVec::size());
        }
        for (; d < size; d++) {
          float grad_val = float(grad_ptr[d]); // + param_ptr[d] * weight_decay;
          exp_avg_ptr[d] =
              at::BFloat16(float(exp_avg_ptr[d]) * beta1 + grad_val * exp_avg_grad_coefficient);
          exp_avg_sq_ptr[d] = at::BFloat16(float(exp_avg_sq_ptr[d]) * beta2 +
                        grad_val * grad_val * exp_avg_sq_grad_coefficient);

          float demon_val;
          demon_val = std::sqrt(float(exp_avg_sq_ptr[d]) / bias_correction2) + eps;
          param_ptr[d] = param_ptr[d] - step_size * float(exp_avg_ptr[d]) / demon_val;
        }
      });

    return 0;
}

