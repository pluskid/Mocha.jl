CIFAR10
=======

.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 1
   blas_set_num_threads(1)

.. code-block:: text

   17-Nov 23:16:48:INFO:root:002800 :: TRAIN obj-val = 0.81475013
   17-Nov 23:18:53:INFO:root:003000 :: TRAIN obj-val = 0.96854031
   17-Nov 23:19:21:INFO:root:
   17-Nov 23:19:21:INFO:root:## Performance on Validation Set
   17-Nov 23:19:21:INFO:root:---------------------------------------------------------
   17-Nov 23:19:21:INFO:root:  Accuracy (avg over 10000) = 67.3000%
   17-Nov 23:19:21:INFO:root:---------------------------------------------------------
   17-Nov 23:19:21:INFO:root:
   17-Nov 23:21:27:INFO:root:003200 :: TRAIN obj-val = 1.09695852
   17-Nov 23:23:36:INFO:root:003400 :: TRAIN obj-val = 0.98007375
   17-Nov 23:25:49:INFO:root:003600 :: TRAIN obj-val = 0.78248519
   17-Nov 23:28:01:INFO:root:003800 :: TRAIN obj-val = 0.75499558
   17-Nov 23:30:14:INFO:root:004000 :: TRAIN obj-val = 0.77041978
   17-Nov 23:30:45:INFO:root:
   17-Nov 23:30:45:INFO:root:## Performance on Validation Set
   17-Nov 23:30:45:INFO:root:---------------------------------------------------------
   17-Nov 23:30:45:INFO:root:  Accuracy (avg over 10000) = 70.1800%
   17-Nov 23:30:45:INFO:root:---------------------------------------------------------
   17-Nov 23:30:45:INFO:root:
   17-Nov 23:32:59:INFO:root:004200 :: TRAIN obj-val = 0.94838876


.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 16
   blas_set_num_threads(16)
   
.. code-block:: text

   17-Nov 22:24:27:INFO:root:002800 :: TRAIN obj-val = 0.85292178
   17-Nov 22:27:50:INFO:root:003000 :: TRAIN obj-val = 0.88822174
   17-Nov 22:28:31:INFO:root:
   17-Nov 22:28:31:INFO:root:## Performance on Validation Set
   17-Nov 22:28:31:INFO:root:---------------------------------------------------------
   17-Nov 22:28:31:INFO:root:  Accuracy (avg over 10000) = 63.3500%
   17-Nov 22:28:31:INFO:root:---------------------------------------------------------
   17-Nov 22:28:31:INFO:root:
   17-Nov 22:31:58:INFO:root:003200 :: TRAIN obj-val = 1.06246507
   17-Nov 22:35:22:INFO:root:003400 :: TRAIN obj-val = 0.94288993
   17-Nov 22:38:46:INFO:root:003600 :: TRAIN obj-val = 0.84770185
   17-Nov 22:42:12:INFO:root:003800 :: TRAIN obj-val = 0.74366856
   17-Nov 22:45:33:INFO:root:004000 :: TRAIN obj-val = 0.79406691
   17-Nov 22:46:12:INFO:root:
   17-Nov 22:46:12:INFO:root:## Performance on Validation Set
   17-Nov 22:46:12:INFO:root:---------------------------------------------------------
   17-Nov 22:46:12:INFO:root:  Accuracy (avg over 10000) = 67.5700%
   17-Nov 22:46:12:INFO:root:---------------------------------------------------------
   17-Nov 22:46:12:INFO:root:
   17-Nov 22:49:35:INFO:root:004200 :: TRAIN obj-val = 1.02186918
   
.. code-block:: text

   I1117 21:55:18.451865 33463 solver.cpp:403] Iteration 2800, lr = 0.001
   I1117 21:57:18.176666 33463 solver.cpp:247] Iteration 3000, Testing net (#0)
   I1117 21:57:47.454730 33463 solver.cpp:298]     Test net output #0: accuracy = 0.5853
   I1117 21:57:47.454778 33463 solver.cpp:298]     Test net output #1: loss = 1.1544 (* 1 = 1.1544 loss)
   I1117 21:57:48.058338 33463 solver.cpp:191] Iteration 3000, loss = 1.30168
   I1117 21:57:48.058384 33463 solver.cpp:206]     Train net output #0: loss = 1.30168 (* 1 = 1.30168 loss)
   I1117 21:57:48.058395 33463 solver.cpp:403] Iteration 3000, lr = 0.001
   I1117 21:59:48.495744 33463 solver.cpp:191] Iteration 3200, loss = 1.10434
   I1117 21:59:48.495982 33463 solver.cpp:206]     Train net output #0: loss = 1.10434 (* 1 = 1.10434 loss)
   I1117 21:59:48.495995 33463 solver.cpp:403] Iteration 3200, lr = 0.001
   I1117 22:01:48.953501 33463 solver.cpp:191] Iteration 3400, loss = 1.04567
   I1117 22:01:48.953748 33463 solver.cpp:206]     Train net output #0: loss = 1.04567 (* 1 = 1.04567 loss)
   I1117 22:01:48.953762 33463 solver.cpp:403] Iteration 3400, lr = 0.001
   I1117 22:03:49.428063 33463 solver.cpp:191] Iteration 3600, loss = 1.24852
   I1117 22:03:49.428390 33463 solver.cpp:206]     Train net output #0: loss = 1.24852 (* 1 = 1.24852 loss)
   I1117 22:03:49.428403 33463 solver.cpp:403] Iteration 3600, lr = 0.001
   I1117 22:05:49.946528 33463 solver.cpp:191] Iteration 3800, loss = 0.937274
   I1117 22:05:49.946780 33463 solver.cpp:206]     Train net output #0: loss = 0.937274 (* 1 = 0.937274 loss)
   I1117 22:05:49.946794 33463 solver.cpp:403] Iteration 3800, lr = 0.001
   I1117 22:07:49.897718 33463 solver.cpp:247] Iteration 4000, Testing net (#0)
   I1117 22:08:19.291095 33463 solver.cpp:298]     Test net output #0: accuracy = 0.6098
   I1117 22:08:19.291141 33463 solver.cpp:298]     Test net output #1: loss = 1.09563 (* 1 = 1.09563 loss)
   I1117 22:08:19.894783 33463 solver.cpp:191] Iteration 4000, loss = 1.22756
   I1117 22:08:19.894830 33463 solver.cpp:206]     Train net output #0: loss = 1.22756 (* 1 = 1.22756 loss)
   I1117 22:08:19.894841 33463 solver.cpp:403] Iteration 4000, lr = 0.001
   I1117 22:10:20.511523 33463 solver.cpp:191] Iteration 4200, loss = 1.00094
   I1117 22:10:20.511780 33463 solver.cpp:206]     Train net output #0: loss = 1.00094 (* 1 = 1.00094 loss)
   I1117 22:10:20.511791 33463 solver.cpp:403] Iteration 4200, lr = 0.001


