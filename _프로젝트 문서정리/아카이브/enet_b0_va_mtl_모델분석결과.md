(calmman-gpu) PS D:\my_projects\funny_clip_classify> python .\analyze_model.py
==================================================
모델 구조 분석 시작
==================================================
✅ 모델 파일 로드 성공: ./models/affectnet_emotions/enet_b0_8_va_mtl.pt

📋 저장된 데이터 타입: <class 'timm.models.efficientnet.EfficientNet'>
📋 모델 객체로 저장됨

🔍 레이어 구조 분석:
--------------------------------------------------
conv_stem.weight                         | Shape: [32, 3, 3, 3]        | Type: torch.float32
bn1.weight                               | Shape: [32]                 | Type: torch.float32
bn1.bias                                 | Shape: [32]                 | Type: torch.float32
bn1.running_mean                         | Shape: [32]                 | Type: torch.float32
bn1.running_var                          | Shape: [32]                 | Type: torch.float32
bn1.num_batches_tracked                  | Shape: []                   | Type: torch.int64
blocks.0.0.conv_dw.weight                | Shape: [32, 1, 3, 3]        | Type: torch.float32
blocks.0.0.bn1.weight                    | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.bias                      | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.running_mean              | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.running_var               | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.0.0.se.conv_reduce.weight         | Shape: [8, 32, 1, 1]        | Type: torch.float32
blocks.0.0.se.conv_reduce.bias           | Shape: [8]                  | Type: torch.float32
blocks.0.0.se.conv_expand.weight         | Shape: [32, 8, 1, 1]        | Type: torch.float32
blocks.0.0.se.conv_expand.bias           | Shape: [32]                 | Type: torch.float32
blocks.0.0.conv_pw.weight                | Shape: [16, 32, 1, 1]       | Type: torch.float32
blocks.0.0.bn2.weight                    | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.bias                      | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.running_mean              | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.running_var               | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.0.conv_pw.weight                | Shape: [96, 16, 1, 1]       | Type: torch.float32
blocks.1.0.bn1.weight                    | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.bias                      | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.running_mean              | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.running_var               | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.0.conv_dw.weight                | Shape: [96, 1, 3, 3]        | Type: torch.float32
blocks.1.0.bn2.weight                    | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.bias                      | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.running_mean              | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.running_var               | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.0.se.conv_reduce.weight         | Shape: [4, 96, 1, 1]        | Type: torch.float32
blocks.1.0.se.conv_reduce.bias           | Shape: [4]                  | Type: torch.float32
blocks.1.0.se.conv_expand.weight         | Shape: [96, 4, 1, 1]        | Type: torch.float32
blocks.1.0.se.conv_expand.bias           | Shape: [96]                 | Type: torch.float32
blocks.1.0.conv_pwl.weight               | Shape: [24, 96, 1, 1]       | Type: torch.float32
blocks.1.0.bn3.weight                    | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.bias                      | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.running_mean              | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.running_var               | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.1.conv_pw.weight                | Shape: [144, 24, 1, 1]      | Type: torch.float32
blocks.1.1.bn1.weight                    | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.bias                      | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.running_mean              | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.running_var               | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.1.conv_dw.weight                | Shape: [144, 1, 3, 3]       | Type: torch.float32
blocks.1.1.bn2.weight                    | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.bias                      | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.running_mean              | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.running_var               | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.1.se.conv_reduce.weight         | Shape: [6, 144, 1, 1]       | Type: torch.float32
blocks.1.1.se.conv_reduce.bias           | Shape: [6]                  | Type: torch.float32
blocks.1.1.se.conv_expand.weight         | Shape: [144, 6, 1, 1]       | Type: torch.float32
blocks.1.1.se.conv_expand.bias           | Shape: [144]                | Type: torch.float32
blocks.1.1.conv_pwl.weight               | Shape: [24, 144, 1, 1]      | Type: torch.float32
blocks.1.1.bn3.weight                    | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.bias                      | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.running_mean              | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.running_var               | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.0.conv_pw.weight                | Shape: [144, 24, 1, 1]      | Type: torch.float32
blocks.2.0.bn1.weight                    | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.bias                      | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.running_mean              | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.running_var               | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.0.conv_dw.weight                | Shape: [144, 1, 5, 5]       | Type: torch.float32
blocks.2.0.bn2.weight                    | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.bias                      | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.running_mean              | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.running_var               | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.0.se.conv_reduce.weight         | Shape: [6, 144, 1, 1]       | Type: torch.float32
blocks.2.0.se.conv_reduce.bias           | Shape: [6]                  | Type: torch.float32
blocks.2.0.se.conv_expand.weight         | Shape: [144, 6, 1, 1]       | Type: torch.float32
blocks.2.0.se.conv_expand.bias           | Shape: [144]                | Type: torch.float32
blocks.2.0.conv_pwl.weight               | Shape: [40, 144, 1, 1]      | Type: torch.float32
blocks.2.0.bn3.weight                    | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.bias                      | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.running_mean              | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.running_var               | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.1.conv_pw.weight                | Shape: [240, 40, 1, 1]      | Type: torch.float32
blocks.2.1.bn1.weight                    | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.bias                      | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.running_mean              | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.running_var               | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.1.conv_dw.weight                | Shape: [240, 1, 5, 5]       | Type: torch.float32
blocks.2.1.bn2.weight                    | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.bias                      | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.running_mean              | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.running_var               | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.1.se.conv_reduce.weight         | Shape: [10, 240, 1, 1]      | Type: torch.float32
blocks.2.1.se.conv_reduce.bias           | Shape: [10]                 | Type: torch.float32
blocks.2.1.se.conv_expand.weight         | Shape: [240, 10, 1, 1]      | Type: torch.float32
blocks.2.1.se.conv_expand.bias           | Shape: [240]                | Type: torch.float32
blocks.2.1.conv_pwl.weight               | Shape: [40, 240, 1, 1]      | Type: torch.float32
blocks.2.1.bn3.weight                    | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.bias                      | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.running_mean              | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.running_var               | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.0.conv_pw.weight                | Shape: [240, 40, 1, 1]      | Type: torch.float32
blocks.3.0.bn1.weight                    | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.bias                      | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.running_mean              | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.running_var               | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.0.conv_dw.weight                | Shape: [240, 1, 3, 3]       | Type: torch.float32
blocks.3.0.bn2.weight                    | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.bias                      | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.running_mean              | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.running_var               | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.0.se.conv_reduce.weight         | Shape: [10, 240, 1, 1]      | Type: torch.float32
blocks.3.0.se.conv_reduce.bias           | Shape: [10]                 | Type: torch.float32
blocks.3.0.se.conv_expand.weight         | Shape: [240, 10, 1, 1]      | Type: torch.float32
blocks.3.0.se.conv_expand.bias           | Shape: [240]                | Type: torch.float32
blocks.3.0.conv_pwl.weight               | Shape: [80, 240, 1, 1]      | Type: torch.float32
blocks.3.0.bn3.weight                    | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.bias                      | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.running_mean              | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.running_var               | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.1.conv_pw.weight                | Shape: [480, 80, 1, 1]      | Type: torch.float32
blocks.3.1.bn1.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.1.conv_dw.weight                | Shape: [480, 1, 3, 3]       | Type: torch.float32
blocks.3.1.bn2.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.1.se.conv_reduce.weight         | Shape: [20, 480, 1, 1]      | Type: torch.float32
blocks.3.1.se.conv_reduce.bias           | Shape: [20]                 | Type: torch.float32
blocks.3.1.se.conv_expand.weight         | Shape: [480, 20, 1, 1]      | Type: torch.float32
blocks.3.1.se.conv_expand.bias           | Shape: [480]                | Type: torch.float32
blocks.3.1.conv_pwl.weight               | Shape: [80, 480, 1, 1]      | Type: torch.float32
blocks.3.1.bn3.weight                    | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.bias                      | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.running_mean              | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.running_var               | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.2.conv_pw.weight                | Shape: [480, 80, 1, 1]      | Type: torch.float32
blocks.3.2.bn1.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.2.conv_dw.weight                | Shape: [480, 1, 3, 3]       | Type: torch.float32
blocks.3.2.bn2.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.2.se.conv_reduce.weight         | Shape: [20, 480, 1, 1]      | Type: torch.float32
blocks.3.2.se.conv_reduce.bias           | Shape: [20]                 | Type: torch.float32
blocks.3.2.se.conv_expand.weight         | Shape: [480, 20, 1, 1]      | Type: torch.float32
blocks.3.2.se.conv_expand.bias           | Shape: [480]                | Type: torch.float32
blocks.3.2.conv_pwl.weight               | Shape: [80, 480, 1, 1]      | Type: torch.float32
blocks.3.2.bn3.weight                    | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.bias                      | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.running_mean              | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.running_var               | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.0.conv_pw.weight                | Shape: [480, 80, 1, 1]      | Type: torch.float32
blocks.4.0.bn1.weight                    | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.bias                      | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.running_mean              | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.running_var               | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.0.conv_dw.weight                | Shape: [480, 1, 5, 5]       | Type: torch.float32
blocks.4.0.bn2.weight                    | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.bias                      | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.running_mean              | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.running_var               | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.0.se.conv_reduce.weight         | Shape: [20, 480, 1, 1]      | Type: torch.float32
blocks.4.0.se.conv_reduce.bias           | Shape: [20]                 | Type: torch.float32
blocks.4.0.se.conv_expand.weight         | Shape: [480, 20, 1, 1]      | Type: torch.float32
blocks.4.0.se.conv_expand.bias           | Shape: [480]                | Type: torch.float32
blocks.4.0.conv_pwl.weight               | Shape: [112, 480, 1, 1]     | Type: torch.float32
blocks.4.0.bn3.weight                    | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.bias                      | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.running_mean              | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.running_var               | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.1.conv_pw.weight                | Shape: [672, 112, 1, 1]     | Type: torch.float32
blocks.4.1.bn1.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.1.conv_dw.weight                | Shape: [672, 1, 5, 5]       | Type: torch.float32
blocks.4.1.bn2.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.1.se.conv_reduce.weight         | Shape: [28, 672, 1, 1]      | Type: torch.float32
blocks.4.1.se.conv_reduce.bias           | Shape: [28]                 | Type: torch.float32
blocks.4.1.se.conv_expand.weight         | Shape: [672, 28, 1, 1]      | Type: torch.float32
blocks.4.1.se.conv_expand.bias           | Shape: [672]                | Type: torch.float32
blocks.4.1.conv_pwl.weight               | Shape: [112, 672, 1, 1]     | Type: torch.float32
blocks.4.1.bn3.weight                    | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.bias                      | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.running_mean              | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.running_var               | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.2.conv_pw.weight                | Shape: [672, 112, 1, 1]     | Type: torch.float32
blocks.4.2.bn1.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.2.conv_dw.weight                | Shape: [672, 1, 5, 5]       | Type: torch.float32
blocks.4.2.bn2.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.2.se.conv_reduce.weight         | Shape: [28, 672, 1, 1]      | Type: torch.float32
blocks.4.2.se.conv_reduce.bias           | Shape: [28]                 | Type: torch.float32
blocks.4.2.se.conv_expand.weight         | Shape: [672, 28, 1, 1]      | Type: torch.float32
blocks.4.2.se.conv_expand.bias           | Shape: [672]                | Type: torch.float32
blocks.4.2.conv_pwl.weight               | Shape: [112, 672, 1, 1]     | Type: torch.float32
blocks.4.2.bn3.weight                    | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.bias                      | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.running_mean              | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.running_var               | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.0.conv_pw.weight                | Shape: [672, 112, 1, 1]     | Type: torch.float32
blocks.5.0.bn1.weight                    | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.bias                      | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.running_mean              | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.running_var               | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.0.conv_dw.weight                | Shape: [672, 1, 5, 5]       | Type: torch.float32
blocks.5.0.bn2.weight                    | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.bias                      | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.running_mean              | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.running_var               | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.0.se.conv_reduce.weight         | Shape: [28, 672, 1, 1]      | Type: torch.float32
blocks.5.0.se.conv_reduce.bias           | Shape: [28]                 | Type: torch.float32
blocks.5.0.se.conv_expand.weight         | Shape: [672, 28, 1, 1]      | Type: torch.float32
blocks.5.0.se.conv_expand.bias           | Shape: [672]                | Type: torch.float32
blocks.5.0.conv_pwl.weight               | Shape: [192, 672, 1, 1]     | Type: torch.float32
blocks.5.0.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.1.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.5.1.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.1.conv_dw.weight                | Shape: [1152, 1, 5, 5]      | Type: torch.float32
blocks.5.1.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.1.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.5.1.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.5.1.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.5.1.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.5.1.conv_pwl.weight               | Shape: [192, 1152, 1, 1]    | Type: torch.float32
blocks.5.1.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.2.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.5.2.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.2.conv_dw.weight                | Shape: [1152, 1, 5, 5]      | Type: torch.float32
blocks.5.2.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.2.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.5.2.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.5.2.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.5.2.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.5.2.conv_pwl.weight               | Shape: [192, 1152, 1, 1]    | Type: torch.float32
blocks.5.2.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.3.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.5.3.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.3.conv_dw.weight                | Shape: [1152, 1, 5, 5]      | Type: torch.float32
blocks.5.3.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.3.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.5.3.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.5.3.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.5.3.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.5.3.conv_pwl.weight               | Shape: [192, 1152, 1, 1]    | Type: torch.float32
blocks.5.3.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.6.0.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.6.0.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.6.0.conv_dw.weight                | Shape: [1152, 1, 3, 3]      | Type: torch.float32
blocks.6.0.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.6.0.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.6.0.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.6.0.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.6.0.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.6.0.conv_pwl.weight               | Shape: [320, 1152, 1, 1]    | Type: torch.float32
blocks.6.0.bn3.weight                    | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.bias                      | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.running_mean              | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.running_var               | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
conv_head.weight                         | Shape: [1280, 320, 1, 1]    | Type: torch.float32
bn2.weight                               | Shape: [1280]               | Type: torch.float32
bn2.bias                                 | Shape: [1280]               | Type: torch.float32
bn2.running_mean                         | Shape: [1280]               | Type: torch.float32
bn2.running_var                          | Shape: [1280]               | Type: torch.float32
bn2.num_batches_tracked                  | Shape: []                   | Type: torch.int64
classifier.weight                        | Shape: [10, 1280]           | Type: torch.float32
classifier.bias                          | Shape: [10]                 | Type: torch.float32

🎯 특징 추출 관련 레이어 찾기:
--------------------------------------------------
🔧 특징 추출 레이어들:

🎯 분류 레이어들:
  conv_head.weight                         | torch.Size([1280, 320, 1, 1])
  classifier.weight                        | torch.Size([10, 1280])
  classifier.bias                          | torch.Size([10])

📐 특징 벡터 차원 추정:
--------------------------------------------------
📏 conv_head.weight → 입력 차원: 1
📏 classifier.weight → 입력 차원: 1280

📊 요약 정보:
--------------------------------------------------
총 파라미터 수: 4,062,423
레이어 수: 360
추천 특징 벡터 차원: 1280

🛠️ 특징 추출기 생성:
------------------------------
==================================================
모델 구조 분석 시작
==================================================
✅ 모델 파일 로드 성공: ./models/affectnet_emotions/enet_b0_8_va_mtl.pt

📋 저장된 데이터 타입: <class 'timm.models.efficientnet.EfficientNet'>
📋 모델 객체로 저장됨

🔍 레이어 구조 분석:
--------------------------------------------------
conv_stem.weight                         | Shape: [32, 3, 3, 3]        | Type: torch.float32
bn1.weight                               | Shape: [32]                 | Type: torch.float32
bn1.bias                                 | Shape: [32]                 | Type: torch.float32
bn1.running_mean                         | Shape: [32]                 | Type: torch.float32
bn1.running_var                          | Shape: [32]                 | Type: torch.float32
bn1.num_batches_tracked                  | Shape: []                   | Type: torch.int64
blocks.0.0.conv_dw.weight                | Shape: [32, 1, 3, 3]        | Type: torch.float32
blocks.0.0.bn1.weight                    | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.bias                      | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.running_mean              | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.running_var               | Shape: [32]                 | Type: torch.float32
blocks.0.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.0.0.se.conv_reduce.weight         | Shape: [8, 32, 1, 1]        | Type: torch.float32
blocks.0.0.se.conv_reduce.bias           | Shape: [8]                  | Type: torch.float32
blocks.0.0.se.conv_expand.weight         | Shape: [32, 8, 1, 1]        | Type: torch.float32
blocks.0.0.se.conv_expand.bias           | Shape: [32]                 | Type: torch.float32
blocks.0.0.conv_pw.weight                | Shape: [16, 32, 1, 1]       | Type: torch.float32
blocks.0.0.bn2.weight                    | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.bias                      | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.running_mean              | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.running_var               | Shape: [16]                 | Type: torch.float32
blocks.0.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.0.conv_pw.weight                | Shape: [96, 16, 1, 1]       | Type: torch.float32
blocks.1.0.bn1.weight                    | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.bias                      | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.running_mean              | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.running_var               | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.0.conv_dw.weight                | Shape: [96, 1, 3, 3]        | Type: torch.float32
blocks.1.0.bn2.weight                    | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.bias                      | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.running_mean              | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.running_var               | Shape: [96]                 | Type: torch.float32
blocks.1.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.0.se.conv_reduce.weight         | Shape: [4, 96, 1, 1]        | Type: torch.float32
blocks.1.0.se.conv_reduce.bias           | Shape: [4]                  | Type: torch.float32
blocks.1.0.se.conv_expand.weight         | Shape: [96, 4, 1, 1]        | Type: torch.float32
blocks.1.0.se.conv_expand.bias           | Shape: [96]                 | Type: torch.float32
blocks.1.0.conv_pwl.weight               | Shape: [24, 96, 1, 1]       | Type: torch.float32
blocks.1.0.bn3.weight                    | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.bias                      | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.running_mean              | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.running_var               | Shape: [24]                 | Type: torch.float32
blocks.1.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.1.conv_pw.weight                | Shape: [144, 24, 1, 1]      | Type: torch.float32
blocks.1.1.bn1.weight                    | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.bias                      | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.running_mean              | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.running_var               | Shape: [144]                | Type: torch.float32
blocks.1.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.1.conv_dw.weight                | Shape: [144, 1, 3, 3]       | Type: torch.float32
blocks.1.1.bn2.weight                    | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.bias                      | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.running_mean              | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.running_var               | Shape: [144]                | Type: torch.float32
blocks.1.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.1.1.se.conv_reduce.weight         | Shape: [6, 144, 1, 1]       | Type: torch.float32
blocks.1.1.se.conv_reduce.bias           | Shape: [6]                  | Type: torch.float32
blocks.1.1.se.conv_expand.weight         | Shape: [144, 6, 1, 1]       | Type: torch.float32
blocks.1.1.se.conv_expand.bias           | Shape: [144]                | Type: torch.float32
blocks.1.1.conv_pwl.weight               | Shape: [24, 144, 1, 1]      | Type: torch.float32
blocks.1.1.bn3.weight                    | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.bias                      | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.running_mean              | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.running_var               | Shape: [24]                 | Type: torch.float32
blocks.1.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.0.conv_pw.weight                | Shape: [144, 24, 1, 1]      | Type: torch.float32
blocks.2.0.bn1.weight                    | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.bias                      | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.running_mean              | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.running_var               | Shape: [144]                | Type: torch.float32
blocks.2.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.0.conv_dw.weight                | Shape: [144, 1, 5, 5]       | Type: torch.float32
blocks.2.0.bn2.weight                    | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.bias                      | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.running_mean              | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.running_var               | Shape: [144]                | Type: torch.float32
blocks.2.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.0.se.conv_reduce.weight         | Shape: [6, 144, 1, 1]       | Type: torch.float32
blocks.2.0.se.conv_reduce.bias           | Shape: [6]                  | Type: torch.float32
blocks.2.0.se.conv_expand.weight         | Shape: [144, 6, 1, 1]       | Type: torch.float32
blocks.2.0.se.conv_expand.bias           | Shape: [144]                | Type: torch.float32
blocks.2.0.conv_pwl.weight               | Shape: [40, 144, 1, 1]      | Type: torch.float32
blocks.2.0.bn3.weight                    | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.bias                      | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.running_mean              | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.running_var               | Shape: [40]                 | Type: torch.float32
blocks.2.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.1.conv_pw.weight                | Shape: [240, 40, 1, 1]      | Type: torch.float32
blocks.2.1.bn1.weight                    | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.bias                      | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.running_mean              | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.running_var               | Shape: [240]                | Type: torch.float32
blocks.2.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.1.conv_dw.weight                | Shape: [240, 1, 5, 5]       | Type: torch.float32
blocks.2.1.bn2.weight                    | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.bias                      | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.running_mean              | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.running_var               | Shape: [240]                | Type: torch.float32
blocks.2.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.2.1.se.conv_reduce.weight         | Shape: [10, 240, 1, 1]      | Type: torch.float32
blocks.2.1.se.conv_reduce.bias           | Shape: [10]                 | Type: torch.float32
blocks.2.1.se.conv_expand.weight         | Shape: [240, 10, 1, 1]      | Type: torch.float32
blocks.2.1.se.conv_expand.bias           | Shape: [240]                | Type: torch.float32
blocks.2.1.conv_pwl.weight               | Shape: [40, 240, 1, 1]      | Type: torch.float32
blocks.2.1.bn3.weight                    | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.bias                      | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.running_mean              | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.running_var               | Shape: [40]                 | Type: torch.float32
blocks.2.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.0.conv_pw.weight                | Shape: [240, 40, 1, 1]      | Type: torch.float32
blocks.3.0.bn1.weight                    | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.bias                      | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.running_mean              | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.running_var               | Shape: [240]                | Type: torch.float32
blocks.3.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.0.conv_dw.weight                | Shape: [240, 1, 3, 3]       | Type: torch.float32
blocks.3.0.bn2.weight                    | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.bias                      | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.running_mean              | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.running_var               | Shape: [240]                | Type: torch.float32
blocks.3.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.0.se.conv_reduce.weight         | Shape: [10, 240, 1, 1]      | Type: torch.float32
blocks.3.0.se.conv_reduce.bias           | Shape: [10]                 | Type: torch.float32
blocks.3.0.se.conv_expand.weight         | Shape: [240, 10, 1, 1]      | Type: torch.float32
blocks.3.0.se.conv_expand.bias           | Shape: [240]                | Type: torch.float32
blocks.3.0.conv_pwl.weight               | Shape: [80, 240, 1, 1]      | Type: torch.float32
blocks.3.0.bn3.weight                    | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.bias                      | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.running_mean              | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.running_var               | Shape: [80]                 | Type: torch.float32
blocks.3.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.1.conv_pw.weight                | Shape: [480, 80, 1, 1]      | Type: torch.float32
blocks.3.1.bn1.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.1.conv_dw.weight                | Shape: [480, 1, 3, 3]       | Type: torch.float32
blocks.3.1.bn2.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.1.se.conv_reduce.weight         | Shape: [20, 480, 1, 1]      | Type: torch.float32
blocks.3.1.se.conv_reduce.bias           | Shape: [20]                 | Type: torch.float32
blocks.3.1.se.conv_expand.weight         | Shape: [480, 20, 1, 1]      | Type: torch.float32
blocks.3.1.se.conv_expand.bias           | Shape: [480]                | Type: torch.float32
blocks.3.1.conv_pwl.weight               | Shape: [80, 480, 1, 1]      | Type: torch.float32
blocks.3.1.bn3.weight                    | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.bias                      | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.running_mean              | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.running_var               | Shape: [80]                 | Type: torch.float32
blocks.3.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.2.conv_pw.weight                | Shape: [480, 80, 1, 1]      | Type: torch.float32
blocks.3.2.bn1.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.2.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.2.conv_dw.weight                | Shape: [480, 1, 3, 3]       | Type: torch.float32
blocks.3.2.bn2.weight                    | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.bias                      | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.running_mean              | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.running_var               | Shape: [480]                | Type: torch.float32
blocks.3.2.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.3.2.se.conv_reduce.weight         | Shape: [20, 480, 1, 1]      | Type: torch.float32
blocks.3.2.se.conv_reduce.bias           | Shape: [20]                 | Type: torch.float32
blocks.3.2.se.conv_expand.weight         | Shape: [480, 20, 1, 1]      | Type: torch.float32
blocks.3.2.se.conv_expand.bias           | Shape: [480]                | Type: torch.float32
blocks.3.2.conv_pwl.weight               | Shape: [80, 480, 1, 1]      | Type: torch.float32
blocks.3.2.bn3.weight                    | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.bias                      | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.running_mean              | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.running_var               | Shape: [80]                 | Type: torch.float32
blocks.3.2.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.0.conv_pw.weight                | Shape: [480, 80, 1, 1]      | Type: torch.float32
blocks.4.0.bn1.weight                    | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.bias                      | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.running_mean              | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.running_var               | Shape: [480]                | Type: torch.float32
blocks.4.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.0.conv_dw.weight                | Shape: [480, 1, 5, 5]       | Type: torch.float32
blocks.4.0.bn2.weight                    | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.bias                      | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.running_mean              | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.running_var               | Shape: [480]                | Type: torch.float32
blocks.4.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.0.se.conv_reduce.weight         | Shape: [20, 480, 1, 1]      | Type: torch.float32
blocks.4.0.se.conv_reduce.bias           | Shape: [20]                 | Type: torch.float32
blocks.4.0.se.conv_expand.weight         | Shape: [480, 20, 1, 1]      | Type: torch.float32
blocks.4.0.se.conv_expand.bias           | Shape: [480]                | Type: torch.float32
blocks.4.0.conv_pwl.weight               | Shape: [112, 480, 1, 1]     | Type: torch.float32
blocks.4.0.bn3.weight                    | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.bias                      | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.running_mean              | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.running_var               | Shape: [112]                | Type: torch.float32
blocks.4.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.1.conv_pw.weight                | Shape: [672, 112, 1, 1]     | Type: torch.float32
blocks.4.1.bn1.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.1.conv_dw.weight                | Shape: [672, 1, 5, 5]       | Type: torch.float32
blocks.4.1.bn2.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.1.se.conv_reduce.weight         | Shape: [28, 672, 1, 1]      | Type: torch.float32
blocks.4.1.se.conv_reduce.bias           | Shape: [28]                 | Type: torch.float32
blocks.4.1.se.conv_expand.weight         | Shape: [672, 28, 1, 1]      | Type: torch.float32
blocks.4.1.se.conv_expand.bias           | Shape: [672]                | Type: torch.float32
blocks.4.1.conv_pwl.weight               | Shape: [112, 672, 1, 1]     | Type: torch.float32
blocks.4.1.bn3.weight                    | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.bias                      | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.running_mean              | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.running_var               | Shape: [112]                | Type: torch.float32
blocks.4.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.2.conv_pw.weight                | Shape: [672, 112, 1, 1]     | Type: torch.float32
blocks.4.2.bn1.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.2.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.2.conv_dw.weight                | Shape: [672, 1, 5, 5]       | Type: torch.float32
blocks.4.2.bn2.weight                    | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.bias                      | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.running_mean              | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.running_var               | Shape: [672]                | Type: torch.float32
blocks.4.2.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.4.2.se.conv_reduce.weight         | Shape: [28, 672, 1, 1]      | Type: torch.float32
blocks.4.2.se.conv_reduce.bias           | Shape: [28]                 | Type: torch.float32
blocks.4.2.se.conv_expand.weight         | Shape: [672, 28, 1, 1]      | Type: torch.float32
blocks.4.2.se.conv_expand.bias           | Shape: [672]                | Type: torch.float32
blocks.4.2.conv_pwl.weight               | Shape: [112, 672, 1, 1]     | Type: torch.float32
blocks.4.2.bn3.weight                    | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.bias                      | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.running_mean              | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.running_var               | Shape: [112]                | Type: torch.float32
blocks.4.2.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.0.conv_pw.weight                | Shape: [672, 112, 1, 1]     | Type: torch.float32
blocks.5.0.bn1.weight                    | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.bias                      | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.running_mean              | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.running_var               | Shape: [672]                | Type: torch.float32
blocks.5.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.0.conv_dw.weight                | Shape: [672, 1, 5, 5]       | Type: torch.float32
blocks.5.0.bn2.weight                    | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.bias                      | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.running_mean              | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.running_var               | Shape: [672]                | Type: torch.float32
blocks.5.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.0.se.conv_reduce.weight         | Shape: [28, 672, 1, 1]      | Type: torch.float32
blocks.5.0.se.conv_reduce.bias           | Shape: [28]                 | Type: torch.float32
blocks.5.0.se.conv_expand.weight         | Shape: [672, 28, 1, 1]      | Type: torch.float32
blocks.5.0.se.conv_expand.bias           | Shape: [672]                | Type: torch.float32
blocks.5.0.conv_pwl.weight               | Shape: [192, 672, 1, 1]     | Type: torch.float32
blocks.5.0.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.1.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.5.1.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.1.conv_dw.weight                | Shape: [1152, 1, 5, 5]      | Type: torch.float32
blocks.5.1.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.1.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.1.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.5.1.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.5.1.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.5.1.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.5.1.conv_pwl.weight               | Shape: [192, 1152, 1, 1]    | Type: torch.float32
blocks.5.1.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.1.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.2.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.5.2.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.2.conv_dw.weight                | Shape: [1152, 1, 5, 5]      | Type: torch.float32
blocks.5.2.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.2.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.2.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.5.2.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.5.2.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.5.2.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.5.2.conv_pwl.weight               | Shape: [192, 1152, 1, 1]    | Type: torch.float32
blocks.5.2.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.2.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.3.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.5.3.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.3.conv_dw.weight                | Shape: [1152, 1, 5, 5]      | Type: torch.float32
blocks.5.3.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.5.3.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.5.3.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.5.3.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.5.3.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.5.3.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.5.3.conv_pwl.weight               | Shape: [192, 1152, 1, 1]    | Type: torch.float32
blocks.5.3.bn3.weight                    | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.bias                      | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.running_mean              | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.running_var               | Shape: [192]                | Type: torch.float32
blocks.5.3.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.6.0.conv_pw.weight                | Shape: [1152, 192, 1, 1]    | Type: torch.float32
blocks.6.0.bn1.weight                    | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.bias                      | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.running_var               | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn1.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.6.0.conv_dw.weight                | Shape: [1152, 1, 3, 3]      | Type: torch.float32
blocks.6.0.bn2.weight                    | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.bias                      | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.running_mean              | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.running_var               | Shape: [1152]               | Type: torch.float32
blocks.6.0.bn2.num_batches_tracked       | Shape: []                   | Type: torch.int64
blocks.6.0.se.conv_reduce.weight         | Shape: [48, 1152, 1, 1]     | Type: torch.float32
blocks.6.0.se.conv_reduce.bias           | Shape: [48]                 | Type: torch.float32
blocks.6.0.se.conv_expand.weight         | Shape: [1152, 48, 1, 1]     | Type: torch.float32
blocks.6.0.se.conv_expand.bias           | Shape: [1152]               | Type: torch.float32
blocks.6.0.conv_pwl.weight               | Shape: [320, 1152, 1, 1]    | Type: torch.float32
blocks.6.0.bn3.weight                    | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.bias                      | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.running_mean              | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.running_var               | Shape: [320]                | Type: torch.float32
blocks.6.0.bn3.num_batches_tracked       | Shape: []                   | Type: torch.int64
conv_head.weight                         | Shape: [1280, 320, 1, 1]    | Type: torch.float32
bn2.weight                               | Shape: [1280]               | Type: torch.float32
bn2.bias                                 | Shape: [1280]               | Type: torch.float32
bn2.running_mean                         | Shape: [1280]               | Type: torch.float32
bn2.running_var                          | Shape: [1280]               | Type: torch.float32
bn2.num_batches_tracked                  | Shape: []                   | Type: torch.int64
classifier.weight                        | Shape: [10, 1280]           | Type: torch.float32
classifier.bias                          | Shape: [10]                 | Type: torch.float32

🎯 특징 추출 관련 레이어 찾기:
--------------------------------------------------
🔧 특징 추출 레이어들:

🎯 분류 레이어들:
  conv_head.weight                         | torch.Size([1280, 320, 1, 1])
  classifier.weight                        | torch.Size([10, 1280])
  classifier.bias                          | torch.Size([10])

📐 특징 벡터 차원 추정:
--------------------------------------------------
📏 conv_head.weight → 입력 차원: 1
📏 classifier.weight → 입력 차원: 1280

📊 요약 정보:
--------------------------------------------------
총 파라미터 수: 4,062,423
레이어 수: 360
추천 특징 벡터 차원: 1280
📦 timm 라이브러리 발견
✅ timm EfficientNet 특징 추출기 생성 완료

🧪 특징 추출 테스트:
------------------------------
✅ 입력 크기: (1, 3, 224, 224)
✅ 출력 특징 크기: torch.Size([1, 1280])
✅ 특징 벡터 차원: 1280