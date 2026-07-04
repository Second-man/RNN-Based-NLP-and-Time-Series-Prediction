###  **提供的数据集** 
task3: 34%
task4: 52%


### **fra.txt数据集**
描述

总句对数: 240521
长度>=2的有效句对: 240005

英文唯一词数量: 19976
法语唯一词数量: 25894
英法合并唯一词数量: 43365

英文平均长度: 6.08 个词
法语平均长度: 7.13 个词

英文最长句: 55 个词
法语最长句: 60 个词

EN_to_FR

没有单独列出来就和第一次一样

**第1次  task3**
dropoupt=0.2
    parser.add_argument("--max_samples", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--max_encoder_len", type=int, default=18)
    parser.add_argument("--max_decoder_len", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=192)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)

    Epoch 18/25
    1125/1125 ━━━━━━━━━━━━━━━━━━━━ 48s 42ms/step - accuracy: 0.7819 - loss: 0.8428 - val_accuracy: 0.6223 - val_loss: 1.9228
    Validation loss: 1.9155
    Validation accuracy: 0.6136

第2次  task3
parser.add_argument("--max_samples", type=int, default=100000)

Epoch 20/25
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 68s 48ms/step - accuracy: 0.8053 - loss: 0.7255 - val_accuracy: 0.6443 - val_loss: 1.7996
Validation loss: 1.7736
Validation accuracy: 0.6407

第3次  task3
parser.add_argument("--max_samples", type=int, default=100000)
parser.add_argument("--max_vocab", type=int, default=15000)
Epoch 18/25
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 88s 62ms/step - accuracy: 0.8019 - loss: 0.7495 - val_accuracy: 0.6367 - val_loss: 1.9107
Validation loss: 1.8875
Validation accuracy: 0.6291

第4次 task3
parser.add_argument("--max_samples", type=int, default=100000)
parser.add_argument("--max_vocab", type=int, default=8000)

Epoch 19/25
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 60s 43ms/step - accuracy: 0.7888 - loss: 0.7919 - val_accuracy: 0.6463 - val_loss: 1.7358

第5次 task3
parser.add_argument("--max_samples", type=int, default=150000)
parser.add_argument("--max_vocab", type=int, default=8000)

Epoch 13/25
2110/2110 ━━━━━━━━━━━━━━━━━━━━ 89s 42ms/step - accuracy: 0.7577 - loss: 0.9254 - val_accuracy: 0.6708 - val_loss: 1.4895

第6次 task3
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=8000)

Epoch 23/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 29s 41ms/step - accuracy: 0.7951 - loss: 0.8053 - val_accuracy: 0.5820 - val_loss: 2.2059
Validation loss: 2.1833
Validation accuracy: 0.5779

第7次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=5000)

Epoch 22/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 21s 30ms/step - accuracy: 0.7692 - loss: 0.8981 - val_accuracy: 0.5943 - val_loss: 1.9906
Validation loss: 1.9751
Validation accuracy: 0.5899

第8次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=3000)

Epoch 23/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 16s 23ms/step - accuracy: 0.7650 - loss: 0.8891 - val_accuracy: 0.6123 - val_loss: 1.7956
Validation loss: 1.7788
Validation accuracy: 0.6102

第9次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=100)

Epoch 16/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 14s 20ms/step - accuracy: 0.7234 - loss: 0.9250 - val_accuracy: 0.6699 - val_loss: 1.1709
Validation loss: 1.1644
Validation accuracy: 0.6689

修改unk使其忽略
第10次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=100)

Epoch 16/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 14s 20ms/step - loss: 0.9205 - masked_accuracy: 0.6360 - val_loss: 1.1708 - val_masked_accuracy: 0.5839
Validation loss: 1.1600
Validation masked accuracy: 0.5779

第11次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=5000)

Epoch 25/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 16s 22ms/step - loss: 0.8203 - masked_accuracy: 0.7897 - val_loss: 2.0479 - val_masked_accuracy: 0.6016
Validation loss: 2.0228
Validation masked accuracy: 0.5986

第12次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=8000)

Epoch 22/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 23s 33ms/step - loss: 0.8469 - masked_accuracy: 0.7888 - val_loss: 2.1998 - val_masked_accuracy: 0.5928
Validation loss: 2.1734
Validation masked accuracy: 0.5877

第13次
parser.add_argument("--max_samples", type=int, default=50000)
parser.add_argument("--max_vocab", type=int, default=5000)
parser.add_argument("--max_encoder_len", type=int, default=8)

Epoch 24/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 17s 25ms/step - loss: 0.6676 - masked_accuracy: 0.8272 - val_loss: 1.8557 - val_masked_accuracy: 0.6390
Validation loss: 1.8181
Validation masked accuracy: 0.6364

第14次
parser.add_argument("--max_samples", type=int, default=150000)
parser.add_argument("--max_vocab", type=int, default=8000)

Epoch 19/25
2110/2110 ━━━━━━━━━━━━━━━━━━━━ 70s 33ms/step - loss: 0.7513 - masked_accuracy: 0.7983 - val_loss: 1.5007 - val_masked_accuracy: 0.6859
Validation loss: 1.4863
Validation masked accuracy: 0.6827

第15次
dropoupt=0.01/50000/8000

Epoch 19/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 23s 33ms/step - loss: 0.8527 - masked_accuracy: 0.7977 - val_loss: 2.4183 - val_masked_accuracy: 0.5619
Validation loss: 2.3485
Validation masked accuracy: 0.5609

第16次
dropoupt=0.1/50000/8000

Epoch 19/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 23s 33ms/step - loss: 0.8984 - masked_accuracy: 0.7807 - val_loss: 2.2645 - val_masked_accuracy: 0.5814
Validation loss: 2.2475
Validation masked accuracy: 0.5714

第17次
dropoupt=0.5

Epoch 25/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 22s 32ms/step - loss: 1.1366 - masked_accuracy: 0.7156 - val_loss: 2.1551 - val_masked_accuracy: 0.5913
Validation loss: 2.1509
Validation masked accuracy: 0.5908

第18次

dropoupt=0.3

Epoch 25/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 24s 34ms/step - loss: 0.8639 - masked_accuracy: 0.7815 - val_loss: 2.1870 - val_masked_accuracy: 0.5977
Validation loss: 2.1754
Validation masked accuracy: 0.5895

—————————————————————————————————————————————————————————————————————————————————————————————————————————————————
第1次 task4
dropout=0.2
parser.add_argument("--max_samples", type=int, default=150000)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_vocab", type=int, default=8000)
parser.add_argument("--max_encoder_len", type=int, default=18)
parser.add_argument("--max_decoder_len", type=int, default=20)
parser.add_argument("--embedding_dim", type=int, default=192)
parser.add_argument("--latent_dim", type=int, default=192)
parser.add_argument("--seed", type=int, default=42)

Epoch 13/25
2110/2110 ━━━━━━━━━━━━━━━━━━━━ 145s 69ms/step - accuracy: 0.8345 - loss: 0.5672 - val_accuracy: 0.7461 - val_loss: 1.2117

第2次 task4
parser.add_argument("--max_samples", type=int, default=50000)

Epoch 25/25
704/704 ━━━━━━━━━━━━━━━━━━━━ 42s 60ms/step - loss: 0.3757 - masked_accuracy: 0.8889 - val_loss: 2.2363 - val_masked_accuracy: 0.6389
Validation loss: 2.2350
Validation masked accuracy: 0.6389

第3次
parser.add_argument("--max_samples", type=int, default=240000)
parser.add_argument("--epochs", type=int, default=50)

Epoch 39/50
3365/3365 ━━━━━━━━━━━━━━━━━━━━ 226s 67ms/step - loss: 0.4413 - masked_accuracy: 0.8639 - val_loss: 1.1169 - val_masked_accuracy: 0.7825
Epoch 40/50
3365/3365 ━━━━━━━━━━━━━━━━━━━━ 226s 67ms/step - loss: 0.4398 - masked_accuracy: 0.8642 - val_loss: 1.1193 - val_masked_accuracy: 0.7813

第4次
dropout=0.3/150000/8000
