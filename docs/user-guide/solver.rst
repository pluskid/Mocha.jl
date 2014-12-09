Solvers
=======

Mocha contains general purpose stochastic (sub-)gradient based solvers that
could be used to train deep neural networks as well as traditional shallow
machine learning models.

A solver is constructed by specifying general *solver parameters* that
characterize *learning rate*, *momentum*, and *stop conditions*, etc. and an
*algorithm* that characterize how the parameters are updated in each solver
iteration. The following is an example taken from the `MNIST tutorial
</tutorial/mnist>`_.

.. code-block:: julia

   params = SolverParameters(max_iter=10000, regu_coef=0.0005,
       mom_policy=MomPolicy.Fixed(0.9),
       lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
       load_from=exp_dir)
   solver = SGD(params)

Moreover, it is usually desired to do some short breaks during training
iterations, for example, to print training progress or to save a snapshot of the
trained model to the disk. In Mocha, this is called *coffee breaks* for solvers.

General Solver Parameters
-------------------------

.. class:: SolverParameters

   .. attribute:: max_iter

      Maximum number of iterations to run.

   .. attribute:: regu_coef

      Global regularization coefficient. Used as a global scaling factor for the
      local regularization coefficient of each trainable parameter.

   .. attribute:: lr_policy

      Policy for learning rate. Note this is also a global scaling factor, as
      each trainable parameter also has local learning rate.

   .. attribute:: mom_policy

      Policy for momentum.

   .. attribute:: load_from

      If specified, the solver will try to load trained network before starting
      the solver loop. This property could be

      * The path to a directory: Mocha will try to locate the latest saved
        JLD snapshot in this directory and load it. A mocha snapshot contains
        trained model and the solver state. So the solver loop will continue
        from the saved state instead of re-starting from iteration 0.
      * The path to a particular JLD snapshot file. The same as above except
        that the user control which particular snapshot to load.
      * The path to a HDF5 model file. A HDF5 model file does not contain solver
        state information. So the solver will start from iteration 0, but
        initialize the network from the model saved in the HDF5 file. This could
        be used to fine-tune a trained (relatively) general model on a domain
        specific (maybe smaller) dataset. You can also load HDF5 models
        `exported from external deep learning tools
        </user-guide/tools/import-caffe-model>`_.

Learning Rate Policy
~~~~~~~~~~~~~~~~~~~~

.. class:: LRPolicy.Fixed

   A fixed learning rate.

.. class:: LRPolicy.Step

   Provide learning rate as base_lr * gamma :sup:`floor(iter / stepsize)`. Here
   *base_lr*, *gamma* and *stepsize* are parameters for the policy and *iter* is
   the training iteration.

.. class:: LRPolicy.Exp

   Provide learning rate as base_lr * gamma :sup:`iter`. Here *base_lr* and
   *gamma* are parameters for the policy and *iter* is the training iteration.

.. class:: LRPolicy.Inv

   Provide learning rate as base_lr * (1 + gamma * iter) :sup:`-power`. Here
   *base_lr*, *gamma* and *power* are parameters for the policy and *iter* is
   the training iteration.

.. class:: LRPolicy.Staged

   This policy provide different learning rate policy at different *stages*.
   Stages are specified by number of training iterations. See `the CIFAR-10
   tutorial </tutorial/cifar10>`_ for an example of staged learning rate policy.

Solver Algorithms
-----------------

.. class:: SGD

   Stochastic Gradient Descent with momentum.

.. class:: Nesterov

   Stochastic Nesterov accelerated gradient method.

Solver Coffee Breaks
--------------------
