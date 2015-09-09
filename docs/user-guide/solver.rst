Solvers
=======

Mocha contains general purpose stochastic (sub-)gradient based solvers that
can be used to train deep neural networks as well as traditional shallow
machine learning models.

A solver is constructed by specifying a dictionary of *solver parameters* that
provide necessary configuration: both general settings like *stop conditions*,
and parameters specific to a particular algorithm such as  *momentum policy*.

You then instantiate the *algorithm* that characterizes how the parameters are
updated in each solver iteration. The following is an example taken from the
:doc:`MNIST tutorial </tutorial/mnist>`.

.. code-block:: julia

   method = SGD()
   params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
       mom_policy=MomPolicy.Fixed(0.9),
       lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
       load_from=exp_dir)
   solver = Solver(method, params)

Moreover, it is usually desired to do some short breaks during training
iterations, for example, to print training progress or to save a snapshot of the
trained model to the disk. In Mocha, these are called *coffee breaks* for solvers.

General Solver Parameters
-------------------------

A instance of :class:`SolverParameters` is just a :class:`Dictionary` with :class:`Symbol`
keys. The ``make_solver_parameters`` function helps to construct this, providing default
values suitable for the solver method.

Some parameters apply to all methods:

   .. attribute:: max_iter

      Maximum number of iterations to run.

   .. attribute:: regu_coef

      Global regularization coefficient. Used as a global scaling factor for the
      local regularization coefficient of each trainable parameter.

   .. attribute:: load_from

      If specified, the solver will try to load a trained network before starting
      the solver loop. This parameter can be

      * The path to a directory: Mocha will try to locate the latest saved
        JLD snapshot in this directory and load it. A mocha snapshot contains
        a trained model and the solver state. So the solver loop will continue
        from the saved state instead of re-starting from iteration 0.
      * The path to a particular JLD snapshot file. The same as above except
        that the user controls which particular snapshot to load.
      * The path to a HDF5 model file. A HDF5 model file does not contain solver
        state information. So the solver will start from iteration 0, but
        initialize the network from the model saved in the HDF5 file. This can
        be used to fine-tune a trained (relatively) general model on a domain
        specific (maybe smaller) dataset. You can also load HDF5 models
        :doc:`exported from external deep learning tools
        </user-guide/tools/import-caffe-model>`.

Solver Algorithms
-----------------

The different solver methods are listed below, together with the ``SolverParameters`` arguments particular to them.

.. class:: SGD

   Stochastic Gradient Descent with momentum.

   .. attribute:: lr_policy

      Policy for learning rate. Note that this is also a global scaling factor, as
      each trainable parameter also has a local learning rate.

   .. attribute:: mom_policy

      Policy for momentum.


.. class:: Nesterov

   Stochastic Nesterov accelerated gradient method.

   .. attribute:: lr_policy

      Policy for learning rate, as for SGD.

   .. attribute:: mom_policy

      Policy for momentum, as for SGD.

.. class:: Adam

   As described in `Adam: A Method for Stochastic Optimization <http://arxiv.org/abs/1412.6980>`_.

   (N.B. The Adam solver sets effective learning rates for each parameter individually, so the
   layer local learning rates are ignored in this case.)

   .. attribute:: alpha

      Step size, defaults to 0.001 - this is approximately the largest step any parameter can take in a single iteration.

   .. attribute:: beta1

      Exponential decay factor for 1st order moment estimates, 0<=beta1<1, default 0.9

   .. attribute:: beta2

      Exponential decay factor for 2nd order moment estimates, 0<=beta1<1, default 0.999

   .. attribute:: epsilon

      Affects scaling of the parameter updates for numerical conditioning, default 1-1e-8


Learning Rate Policy
~~~~~~~~~~~~~~~~~~~~

.. class:: LRPolicy.Fixed

   A fixed learning rate.

.. class:: LRPolicy.Step

   Provide the learning rate as base_lr * gamma :sup:`floor(iter / stepsize)`. Here
   *base_lr*, *gamma* and *stepsize* are parameters for the policy and *iter* is
   the training iteration.

.. class:: LRPolicy.Exp

   Provide the learning rate as base_lr * gamma :sup:`iter`. Here *base_lr* and
   *gamma* are parameters for the policy and *iter* is the training iteration.

.. class:: LRPolicy.Inv

   Provide the learning rate as base_lr * (1 + gamma * iter) :sup:`-power`. Here
   *base_lr*, *gamma* and *power* are parameters for the policy and *iter* is
   the training iteration.

.. class:: LRPolicy.Staged

   This policy provides different learning rate policies at different *stages*.
   Stages are specified by number of training iterations. See :doc:`the CIFAR-10
   tutorial </tutorial/cifar10>` for an example of staged learning rate policy.

.. class:: LRPolicy.DecayOnValidation

   This policy starts with a base learning rate. Each time the performance
   on a validation set is computed, the policy will scale the learning rate down
   by a given factor if the validation performance is poorer compared to the one of the
   last snapshot. In this case it also asks the solver to load the latest saved snapshot
   and restart from there.

   Note in order for this policy to function properly, you need to set up both
   :class:`Snapshot` coffee break and :class:`ValidationPerformance` coffee
   break. The policy works by registering a listener on the
   :class:`ValidationPerformance` coffee break. Whenever the performance is
   computed on a validation set, the listener is notified, and it will compare
   the performance with the previous one on records. If the performance decays,
   it will ask the solver to load the previously saved snapshot (saved by the
   :class:`Snapshot` coffee break), and then scale the learning rate down. Per default
   `LRPolicy.DecayOnValidation` considers a lower performance statistic as better,
   however this can be changed by setting the optional argument `higher_better` to `false`.

   A typical setup is to save one snapshot every epoch, and also check the
   performance on the validation set every epoch. So if the performance decays,
   the learning rate is decreased, and the training will restart from the last
   (good) epoch.

   .. code::

      # starts with lr=base_lr, and scale as lr=lr*lr_ratio
      lr_policy=LRPolicy.DecayOnValidation(base_lr,"accuracy-accuracy",lr_ratio)

      validation_performance = ValidationPerformance(test_net)
      add_coffee_break(solver, validation_performance, every_n_epoch=1)

      # register the listener to get notified on performance validation
      setup(params.lr_policy, validation_performance, solver)

Momentum Policy
~~~~~~~~~~~~~~~

.. class:: MomPolicy.Fixed

   Provide fixed momentum.

.. class:: MomPolicy.Step

   Provide the momentum as min(base_mom * gamma :sup:`floor(iter / stepsize)`,
   max_mom). Here *base_mom*, *gamma*, *stepsize* and *max_mom* are policy
   parameters and *iter* is the training iteration.

.. class:: MomPolicy.Linear

   Provide the momentum as min(base_mom + floor(iter / stepsize) * gamma, max_mom).
   Here *base_mom*, *gamma*, *stepsize* and *max_mom* are policy parameters and
   *iter* is the training iteration.

.. class:: MomPolicy.Staged

   This policy provides different momentum policies at different *stages*.
   Stages are specified by number of training iterations. See
   :class:`LRPolicy.Staged`.


Solver Coffee Breaks
--------------------

Training is a very computationally intensive loop of iterations. Being afraid
that the solver might silently go crazy under such heavy load, Mocha provides
the solver opportunities to have a break periodically. During the breaks, the
solver can have a change of mood by, for example, talking to the outside world
about its "mental status". Here is a snippet taken from :doc:`the MNIST tutorial
</tutorial/mnist>`:

.. code-block:: julia

   # report training progress every 100 iterations
   add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

   # save snapshots every 5000 iterations
   add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

We allow the solver to talk about its training progress every 100 iterations,
and save the trained model to a snapshot every 5000 iterations. Alternatively,
coffee breaks can also be specified by ``every_n_epoch``.

Coffee Lounge
~~~~~~~~~~~~~

Coffee lounge is the place for the solver to have coffee breaks. It provides
a storage for a log of the coffee breaks. For example, when the solver talks
about its training progress, the objective function value at each coffee break
will be recorded. That data can be retrieved for inspection or plotting
later.

The default coffee lounge keeps the storage in memory only. If you want to additionally
save the recordings to disk, you can set up the coffee lounge in the
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
   The training objective function value at the current
   iteration is reported by default. You can also call the function with the following
   named parameters in order to customize the output:

   .. attribute:: statistic_names
   A vector of statistic names to print when summarizing the state, e.g. ``[:iter, :obj_val, :learning_rate]``.  The available statistics will depend on the solver method in use.


   Here are a few examples of usage:

   .. code-block:: julia

      #same as original functionality, shows iteration and obj_val by defualt
      TrainingSummary()

      #will only show objective function value
      TrainingSummary([:iter])

      #shows iteration, obj_val, learning_rate, and momentum
      TrainingSummary([:iter, :obj_val, :learning_rate, :momentum])

   Note that the training summary at iteration 0 shows the results before training starts.
   Also, any values that are shown with this method will also be added to the lounge
   using the `update_statistics()` function.

.. class:: Snapshot

   Automatically save solver and model snapshots to a given snapshot directory.
   The snapshot saved at iteration 0 corresponds to the init model (randomly
   initialized via :doc:`initializers </user-guide/initializer>` or loaded from
   existing model file).

.. class:: ValidationPerformance

   Run an epoch over a validation set and report the performance (e.g.
   multiclass classification accuracy). You will need to construct a validation
   network that shares parameters with the training network and provides access to
   the validation dataset. See :doc:`the MNIST tutorial </tutorial/mnist>` for
   a concrete example.
