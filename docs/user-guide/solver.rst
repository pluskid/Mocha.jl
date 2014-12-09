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

Momentum Policy
~~~~~~~~~~~~~~~

.. class:: MomPolicy.Fixed

   Provide fixed momentum.

.. class:: MomPolicy.Step

   Provide momentum as min(base_mom * gamma :sup:`floor(iter / stepsize)`,
   max_mom). Here *base_mom*, *gamma*, *stepsize* and *max_mom* are policy
   parameters and *iter* is the training iteration.

.. class:: MomPolicy.Linear

   Provide momentum as min(base_mom + floor(iter / stepsize) * gamma, max_mom).
   Here *base_mom*, *gamma*, *stepsize* and *max_mom* are policy parameters and
   *iter* is the training iteration.

Solver Algorithms
-----------------

.. class:: SGD

   Stochastic Gradient Descent with momentum.

.. class:: Nesterov

   Stochastic Nesterov accelerated gradient method.

Solver Coffee Breaks
--------------------

Training is a very computationally intensive loop of iterations. Being afraid
that the solver might silently go crazy under such heavy load, Mocha provides
the solver opportunities to have a break periodically. During the breaks, the
solver could have a change of mood by, for example, talking to the outside world
about its "mental status". Here is a snippet taken from `the MNIST tutorial
</tutorial/mnist>`_:

.. code-block:: julia

   # report training progress every 100 iterations
   add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

   # save snapshots every 5000 iterations
   add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

We allow the solver to talk about its training progress every 100 iterations,
and save the trained model to a snapshot every 5000 iterations. Alternatively,
coffee breaks could also be specified by ``every_n_epoch``.

Coffee Lounge
~~~~~~~~~~~~~

Coffee lounge is the place for solver to have coffee breaks. It provide
a storage for a log of the coffee breaks. For example, when the solver talks
about its training progress, the objective function value at each coffee break
will be recorded. Those data could be retrieved for inspection or plotting
later.

The default coffee lounge keeps the storage in memory only. If you want to also
save the recordings to the disk, you could setup the coffee lounge in the
following way:

.. code-block:: julia

   setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld",
       every_n_iter=1000)

This means the recordings will be saved to the specified file every 1000
iterations. There is one extra keyword parameter for setup coffee lounge:
``file_exists``, which should specify a symbol from the following options

``:merge``
  The default. Try to merge with the existing log file. This is useful if, for
  example, you are resuming from an interrupted training process.
``:overwrite``
  Erase the existing log file if any.
``:panic``
  Exit with error if found the log file already exists.

The logs are stored as simple Julia dictionaries. See ``plot_statistics.jl`` in
the ``tools`` directory for an example of how to retrieve and visualize the
saved information.

Built-in Coffee Breaks
~~~~~~~~~~~~~~~~~~~~~~

.. class:: TrainingSummary

   This is a coffee break in which the solver talks about the training summary.
   Currently, only the training objective function value at the current
   iteration is reported. Reporting for other solver status like the current
   learning rate and momentum could be easily added.

   The training summary at iteration 0 shows the results before training starts.

.. class:: Snapshot

   Automatically save solver and model snapshots to a given snapshot directory.
   The snapshot saved at iteration 0 corresponds to the init model (randomly
   initialized via `initializers </user-guide/initializer>`_ or loaded from
   existing model file).

.. class:: ValidationPerformance

   Run an epoch over a validation set and report the performance (e.g.
   multiclass classification accuracy). You will need to construct a validation
   network that shares parameter with the training network and provide access to
   the validation dataset. See `the MNIST tutorial </tutorial/mnist>`_ for
   a concrete example.
