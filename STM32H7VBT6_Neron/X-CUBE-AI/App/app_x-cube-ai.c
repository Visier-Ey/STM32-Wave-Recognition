
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "network.h"
#include "network_data.h"

/* USER CODE BEGIN includes */
   AI_ALIGNED(32)  static float output_data[AI_NETWORK_OUT_1_SIZE];
   AI_ALIGNED(32)  static float input_data[AI_NETWORK_IN_1_SIZE];
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_NETWORK_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_NETWORK_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle network = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
    if (fct)
      printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
             err.type, err.code);
    else
      printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

    do
    {
    } while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_create_and_init");
    return -1;
  }

  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

#if defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_network_run(network, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_network_get_error(network),
        "ai_network_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */


  extern float waveform_fsk[256];
  extern float waveform_sine[256];
  extern float waveform_triangle[256];
  extern float waveform_bpsk[256];
  int acquire_and_process_data()
  {
    for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++)
    {
      // input_data[i] = test_signals[1][i];
      input_data[i] = waveform_sine[i];
    }
    ai_input[0].data = AI_HANDLE_PTR(input_data);
    ai_output[0].data = AI_HANDLE_PTR(output_data);

    return 0;
  }
#include <stdio.h>
  int post_process()
  {
    // 1. 定义类别名称（与模型输出顺序对应�????
    static const char *class_names[4] = {
        "SINE",     // 对应Python索引0
        "TRIANGLE", // 对应Python索引1
        "FSK",      // 对应Python索引2
        "BPSK"      // 对应Python索引3
    };

    // 2. 找出概率�????高的类别
    int predicted_class = 0;
    float max_prob = output_data[0];

    for (int i = 1; i < AI_NETWORK_OUT_1_SIZE; i++)
    {
      if (output_data[i] > max_prob)
      {
        max_prob = output_data[i];
        predicted_class = i;
      }
    }

    // 3. 格式化显示字符串
    printf("Class: %s", class_names[predicted_class]);
    printf("Confidence: %.2f%%", max_prob * 100);

    // 4. 打印调试信息（可选）
    printf("\nClassification Results:\n");
    for (int i = 0; i < AI_NETWORK_OUT_1_SIZE; i++)
    {
      printf("%s: %.2f%%\n", class_names[i], output_data[i] * 100);
    }
    printf("-> Predicted: %s (%.2f%%)\n", class_names[predicted_class], max_prob * 100);

    // 5. 返回预测的类别索�????
    return 0;
  }
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
    printf("\r\nTEMPLATE - initialization\r\n");

    ai_boostrap(data_activations0);

    if (AI_BUFFER_FORMAT(ai_input) != AI_BUFFER_FORMAT_FLOAT || 
    AI_BUFFER_FORMAT(ai_output) != AI_BUFFER_FORMAT_FLOAT) {
    printf("[AI] Error: Model not in float32 format!\r\n");
    } else {
        printf("[AI] Model initialized successfully.\r\n");
    }
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
    int res = -1;

    printf("TEMPLATE - run - main loop\r\n");

    if (network)
    {

      do
      {
        /* 1 - acquire and pre-process input data */
        res = acquire_and_process_data();
        /* 2 - process the data - call inference engine */
        if (res == 0)
          res = ai_run();
        /* 3- post-process the predictions */
        if (res == 0)
          res = post_process();
      } while (res == 0);
    }

    if (res)
    {
      ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
      ai_log_err(err, "Process has FAILED");
    }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
