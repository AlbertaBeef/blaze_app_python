/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vitis/ai/dim_calc.hpp>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
namespace {

struct Pad_t {
  int l;  // left
  int t;  // top
  int r;  // right
  int b;  // bottom
  int f;  // front
  int k;  // back
};

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    mode_ = op->get_attr<std::string>("mode");
    auto paddings = op->get_attr<std::vector<int32_t>>("paddings");
    //std::cout << "[my_pad-fix.cpp] " << "paddings = " << paddings[0] << ", " << paddings[1] << ", " << paddings[2] << ", " << paddings[3] << ", " << paddings[4] << ", " << paddings[5] << ", " << paddings[6] << ", " << paddings[7] << std::endl;
    CHECK_EQ(paddings.size(), 8) << "only support 4d pad current;" << std::endl;
    pad_ = Pad_t{paddings[4], paddings[2], paddings[5], paddings[3], paddings[6], paddings[7]};
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape[0], output_shape[0]);
    CHECK_EQ(input_shape[1] + pad_.t + pad_.b, output_shape[1]);
    CHECK_EQ(input_shape[2] + pad_.l + pad_.r, output_shape[2]);
    CHECK_EQ(input_shape[3] + pad_.f + pad_.k, output_shape[3]);

    auto src_N = input_shape[0];
    auto src_H = input_shape[1];
    auto src_W = input_shape[2];
    auto src_C = input_shape[3];
    //std::cout << "[my_pad-fix.cpp] " << "input N,H,W,C = " << src_N << "," << src_H << "," << src_W << "," << src_C << std::endl;

    auto dst_N = output_shape[0];
    auto dst_H = output_shape[1];
    auto dst_W = output_shape[2];
    auto dst_C = output_shape[3];
    //std::cout << "[my_pad-fix.cpp] " << "output N,H,W,C = " << dst_N << "," << dst_H << "," << dst_W << "," << dst_C << std::endl;
    
    auto src_col_size = input_shape[3];
    auto dst_col_size = output_shape[3];
    //std::cout << "[my_pad-fix.cpp] " << "src_col_size = " << src_col_size << std::endl;
    //std::cout << "[my_pad-fix.cpp] " << "dst_col_size = " << dst_col_size << std::endl;

    auto src_row_size = input_shape[2] * input_shape[3];
    auto dst_row_size = output_shape[2] * output_shape[3];
    //std::cout << "[my_pad-fix.cpp] " << "src_row_size = " << src_row_size << std::endl;
    //std::cout << "[my_pad-fix.cpp] " << "dst_row_size = " << dst_row_size << std::endl;

    auto src_batch_size = input_shape[1] * input_shape[2] * input_shape[3];
    auto dst_batch_size = output_shape[1] * output_shape[2] * output_shape[3];
    //std::cout << "[my_pad-fix.cpp] " << "src_batch_size = " << src_batch_size << std::endl;
    //std::cout << "[my_pad-fix.cpp] " << "dst_batch_size = " << dst_batch_size << std::endl;

    if (pad_.l == 0 && pad_.t == 0 && pad_.r == 0 && pad_.b == 0 && pad_.f == 0 && pad_.k == 0) {
      std::copy_n(input.data, src_batch_size * src_N, output.data);
      return 0;
    }
    
    if (mode_ == "CONSTANT") {
      auto pad_value = 0;
      for (auto i = 0; i < src_N; i++) {
        auto src = input.data + i * src_batch_size;
        auto dst = output.data + i * dst_batch_size;
        // pad top and bottom
        if (pad_.t > 0) {
          std::fill_n(dst, pad_.t * dst_row_size, pad_value);
        }
        if (pad_.b > 0) {
          std::fill_n(dst + (dst_H - pad_.b) * dst_row_size, pad_.b * dst_row_size, pad_value);
        }
        // pad left and right
        if (pad_.l > 0) {
          for (auto h = pad_.t; h < dst_H - pad_.b; h++) {
            auto offset = h * dst_row_size;
            std::fill_n(dst + offset, pad_.l * dst_C, pad_value);
          }
        }
        if (pad_.r > 0) {
          for (auto h = pad_.t; h < dst_H - pad_.b; h++) {
            auto offset = h * dst_row_size + (dst_W - pad_.r) * dst_C;
            std::fill_n(dst + offset, pad_.r * dst_C, pad_value);
          }
        }
        // pad front and back
        if (pad_.f > 0) {
          // just padd the entire batch, then copy in source data
          std::fill_n(dst, dst_batch_size, pad_value);
        }
        if (pad_.k > 0) {
          // just padd the entire batch, then copy in source data
          std::fill_n(dst, dst_batch_size, pad_value);
        }

        // copy source data
        for (auto h = pad_.t; h < dst_H - pad_.b; h++) {
          //auto src_offset = (h - pad_.t) * src_row_size;
          //auto dst_offset = h * dst_row_size + pad_.l * dst_C;
          //std::copy_n(src + src_offset, src_row_size, dst + dst_offset);
          for (auto w = pad_.l; w < dst_W - pad_.r; w++ ) {
            auto src_offset = ((h - pad_.t) * src_row_size) + ((w - pad_.l) * src_col_size);
            auto dst_offset = (h * dst_row_size) + (w * dst_col_size);
            std::copy_n(src + src_offset, src_col_size, dst + dst_offset);
            //if ((h < 3) && (w < 3)) {
            //  std::cout << "COPY src_offset=" << src_offset << " src_col_size=" << src_col_size << " dst_offset=" << dst_offset << std::endl;                
            //}
          }
        }

#if 0
        std::cout << "INPUT[" << i << "] = [" << std::endl;
        for (int h = 0; h < 3/*src_H*/; h++) {
          std::cout << "\t[" << std::endl;
          for (int w = 0; w < 3/*src_W*/; w++) {
            //std::cout << "index=" << (((i * src_H + h) * src_W + w) * src_C + 0) << std::endl;                
            std::cout << "\t\t[ ";
            for (int c = 0; c < 3/*src_C*/; c++) {
              auto in_idx = ((i * src_H + h) * src_W + w) * src_C + c;
              printf("%03d ",(int8_t)input.data[in_idx]);
            }
            std::cout << "... ";            
            for (int c = src_C-3; c < src_C; c++) {
              auto in_idx = ((i * src_H + h) * src_W + w) * src_C + c;
              printf("%03d ",(int8_t)input.data[in_idx]);
            }
            std::cout << "]" << std::endl;
          }
          std::cout << std::endl << "\t]" << std::endl;
        }
        std::cout << "... ";            
        for (int h = src_H-3; h < src_H; h++) {
          std::cout << "\t[" << std::endl;
          for (int w = src_W-3; w < src_W; w++) {
            //std::cout << "index=" << (((i * src_H + h) * src_W + w) * src_C + 0) << std::endl;                
            std::cout << "\t\t[ ";
            for (int c = 0; c < 3/*src_C*/; c++) {
              auto in_idx = ((i * src_H + h) * src_W + w) * src_C + c;
              printf("%03d ",(int8_t)input.data[in_idx]);
            }
            std::cout << "... ";            
            for (int c = src_C-3; c < src_C; c++) {
              auto in_idx = ((i * src_H + h) * src_W + w) * src_C + c;
              printf("%03d ",(int8_t)input.data[in_idx]);
            }
            std::cout << "]" << std::endl;
          }
          std::cout << std::endl << "\t]" << std::endl;
        }
        std::cout << std::endl << "]" << std::endl;
        
        std::cout << "OUTPUT[" << i << "] = [" << std::endl;
        for (int h = 0; h < 3/*dst_H*/; h++) {
          std::cout << "\t[" << std::endl;
          for (int w = 0; w < 3/*dst_W*/; w++) {
            //std::cout << "index=" << (((i * dst_H + h) * dst_W + w) * dst_C + 0) << std::endl;                
            std::cout << "\t\t[ ";
            for (int c = 0; c < 3/*dst_C*/; c++) {
              auto out_idx = ((i * dst_H + h) * dst_W + w) * dst_C + c;
              printf("%03d ",(int8_t)output.data[out_idx]);
            }
            std::cout << "... ";            
            for (int c = src_C-3; c < src_C+3; c++) {
              auto out_idx = ((i * dst_H + h) * dst_W + w) * dst_C + c;
              printf("%03d ",(int8_t)output.data[out_idx]);
            }
            std::cout << "... ";            
            for (int c = dst_C-3; c < dst_C; c++) {
              auto out_idx = ((i * dst_H + h) * dst_W + w) * dst_C + c;
              printf("%03d ",(int8_t)output.data[out_idx]);
            }
            std::cout << "]" << std::endl;
          }
          std::cout << std::endl << "\t]" << std::endl;
        }
        std::cout << "... ";            
        for (int h = dst_H-3; h < dst_H; h++) {
          std::cout << "\t[" << std::endl;
          for (int w = dst_W-3; w < dst_W; w++) {
            //std::cout << "index=" << (((i * dst_H + h) * dst_W + w) * dst_C + 0) << std::endl;                
            std::cout << "\t\t[ ";
            for (int c = 0; c < 3/*dst_C*/; c++) {
              auto out_idx = ((i * dst_H + h) * dst_W + w) * dst_C + c;
              printf("%03d ",(int8_t)output.data[out_idx]);
            }
            std::cout << "... ";            
            for (int c = src_C-3; c < src_C+3; c++) {
              auto out_idx = ((i * dst_H + h) * dst_W + w) * dst_C + c;
              printf("%03d ",(int8_t)output.data[out_idx]);
            }
            std::cout << "... ";            
            for (int c = dst_C-3; c < dst_C; c++) {
              auto out_idx = ((i * dst_H + h) * dst_W + w) * dst_C + c;
              printf("%03d ",(int8_t)output.data[out_idx]);
            }
            std::cout << "]" << std::endl;
          }
          std::cout << std::endl << "\t]" << std::endl;
        }
        std::cout << std::endl << "]" << std::endl;
#endif

      }
    } else if (mode_ == "SYMMETRIC") {
      for (int n = 0; n < output_shape[0]; n++) {
        for (int h = 0; h < output_shape[1]; h++) {
          for (int w = 0; w < output_shape[2]; w++) {
            for (int c = 0; c < output_shape[3]; c++) {
              int h_idx = h - pad_.t;
              int w_idx = w - pad_.l;
              if (h < pad_.t)
                h_idx = pad_.t - 1 - std::min(h, pad_.t - 1);
              else if (h >= input_shape[1] + pad_.t)
                h_idx =
                    input_shape[1] - 1 -
                    std::min(h - pad_.t - input_shape[1], input_shape[1] - 1);
              if (w < pad_.l)
                w_idx = pad_.l - 1 - std::min(w, pad_.l - 1);
              else if (w >= input_shape[2] + pad_.l)
                w_idx =
                    input_shape[2] - 1 -
                    std::min(w - pad_.l - input_shape[2], input_shape[2] - 1);
              auto out_idx = ((n * output_shape[1] + h) * output_shape[2] + w) *
                                 output_shape[3] +
                             c;
              auto in_idx =
                  ((n * input_shape[1] + h_idx) * input_shape[2] + w_idx) *
                      input_shape[3] +
                  c;
              output.data[out_idx] = input.data[in_idx];
            }
          }
        }
      }
    }
    return 0;
  }

 private:
  std::string mode_;
  Pad_t pad_;
};

}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
