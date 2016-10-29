# Mocha on Amazon Web Services (AWS)
The real power of developing deep learning networks is only realized when you can test your ideas and run your models on powerful 
compute platforms that complete training in a fraction of the time it took only a few years ago.  Today state-of-the-art 
machine learning algorithms routinely run on cloud based compute that offers access to cutting edge GPU technology 
for a tiny cost compared to buying the GPU hardware and running it in your personal or company servers.

Amazon Web Services (AWS) is one of the most popular cloud based services that provides access to such powerful computers.  As Mocha
and Julia mature I'm sure that a pre-configured Amazon Machine Image (AMI) will emerge to run Mocha in the cloud. Now, in October
2016, such an AMI does not exist, but even when a pre-configured image for Mocha does become available I highly recommend following
through this tutorial at least once so you understand how cloud resources are provisioned and configured.

We are going to show you how to take the CIFAR-10 example and get it running in the cloud.  Along the way you will learn how to 
interact with AWS and get a broader understanding of cloud architectures in general.

## Signing up for AWS
The first task in getting up and running in AWS is to set up an account.  If you already have an Amazon.com shopping account these same credentials and methods of payment will get you into AWS.  If not, then sign up for an [account here](aws.amazon.com).  There is no charge for signing up for the account.

Usage for AWS is dependent on two factors.  The type of computer you use, called the *instance type* and the hours you use it for.
For this example we are going to provision an instance type called *p2.xlarge* which contains one NVIDIA Tesla K80 GPU, and costs about
90 cents per hour (as of Oct 2016).  Building the software and running the CIFAR10 training will take less than two hours.

**However,** AWS does not let brand new users start-up *p2* or *g2* instances when you first open your account.  So you need to apply
for access to GPU enabled machines by opening a support request.  From the AWS Console (after signing in) click on *support* in the
top right hand corner and select *support center* from the dropdown menu.  When the support page opens up click on ![Create Case](./create_case.png).  This will open up a form similar to the figure below.  Choose the region closest to your location and submit the request with the text and options suggested by the figure.
![Support Request](./support_request.png)

## While you wait
Approval of the support request might take a few days.  So while you wait ![Wink](./smile.png) let me suggest a few ways to
sharpen your knowlege of AWS, Mocha, or deep learning 

#### Track 1 - Deep learning expert learning about AWS

#### Track 2 - Cloud comfortable but new to Mocha or Deep Learning
Since we are going to be using the CIFAR10 example later in this tutorial and training it in the cloud, why not 
download the original paper on the dataset and one of the first convolutional neural network implementations
that threw the gates open to deep learning back in 2012.  Before this seminal paper neural networks deeper than
one or two layers were untrainable because the backpropogation of gradients from the output layer had lost their
corrective strength by the time they reached the bottom layers of deep networks.  

Might have to sign up and wait a few days.
Work through the EC2 tutorials for the few days so you learn how to launch and manage your instances.
Offer reading list suggestions and other tutorials to get done and then come back to this one.

## Provisioning the instance and the base image
At this point you should have heard back from AWS that you are approved for AWS *g2* and *p2* instance types.
Now we need to launch a GPU instance with NVIDIA drivers and the Cuda components needed to work with the GPU.

First let me explain just a little about Cuda.  

The Amazon *p2* instance contains both a CPU and an [NVIDIA Tesla K80](http://www.nvidia.com/object/tesla-servers.html) GPU, but
in order to access the GPU the NVIDIA drivers for the P80 must be installed and the various elements of the NVIDIA development
environment must also be installed.  This tutorial describes how to install these components from scratch, but I find that
installing the correct driver and a compatible version of the development kit that is also compliant with the hardware on the
cloud server can be quite a challenge.


[potential NVIDIA managed AMI](https://aws.amazon.com/marketplace/pp/B01LZMLK1K)
We will launch an AMI

Now we will launch a *p2.xlarge* instance with an AMI that already includes the Cuda components needed to work with the GPU.

## Installing Julia
Julia is new, which means that a lot of things that are annoying about other programming languages are "fixed" in Julia.
Julia is new, which also means that it is not pre-installed in very many AWS machine images (AMIs).  Note that I have tried the

#### Build from scratch in yum based AMI's
```
#! /bin/bash
# Install build environment
# The sed instruction allows the Extended Packages For
# Enterprise Linux (epel) repository to get added to the yum package
# manager. sed -i option edits in place.
echo "*****************Setting up the Build Environment******************"
sudo sed -i 's/enabled=0/enabled=1/' /etc/yum.repos.d/epel.repo
sudo yum -y update
sudo yum -y upgrade
sudo yum -y install git gcc-gfortran clang m4 patch ncurses-devel python-devel

#Set up Julia
echo "*****************Cloning Julia*************************************"
git clone https://github.com/JuliaLang/julia.git
cd julia

#Determine the number of CPUs to build on
NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ {print $2}')

echo "*****************Making Julia on $NUM_CPUS CPUs***************************"
#Takes 30 minutes on a 4CPU p2.xlarge AWS instance
time make -j $NUM_CPUS
```

#### Ubuntu instructions
```
sudo add-apt-repository ppa:staticfloat/juliareleases
sudo add-apt-repository ppa:staticfloat/julia-deps
sudo apt-get update
sudo apt-get install julia
```

* install packages needed to build Julia from source with `sudo apt-get install g++ gfortran m4 cmake pkg-config git hdf5-tools`.  Note that `hdf5-tools` is not required to install Julia, but is required to install the `Mocha` package later in this build guide, and when Julia tries to call `sudo apt-get` from within the REPL it aborts in AWS.  So it is better to get `hdf5-tools` in place now. 
		* Note that the version of Julia available in apt-get is just 0.2 which failed when I tried to run the `cifar10.jl` script.  So clone the git repo with `git clone git://github.com/JuliaLang/julia.git`.  This installs the source in a `/julia` directory.  
		* I built Julia from the head of the master branch the first time and it failed to run `Pkg.test("Mocha")` later in the process.  So it is better practice to build a core component such as a programming language from its stable release unless you plan to contribute to the development of the language itself.  To find a stable version and build against that version we will use the version control properties of `git`.
		* Change directory into the newly cloned julia folder with `cd julia`.  Then issue a `git status` command.  You should see git identifies this folder as a project under version control.  Now issue the `git tag` command.  This will provide a list of tagged releases similar to the list below:
	> v0.4.5  
	> v0.4.6  
	> v0.4.7  
	> v0.5.0  
	> v0.5.0-rc0  
	> v0.5.0-rc1  
		* We want to use the last stable release not a release candidate `v0.X.0-rcX`.  So we issue a git command to checkout to the most recent stable branch which at the time I made this tutorial would be `git checkout v0.5.0`.  
		* We want to build Julia on the maximum number of cores available to the server. To find the number of available cores run `lscpu`.  See the link [here](http://unix.stackexchange.com/questions/218074/how-to-know-number-of-cores-of-a-system-in-linux) for a good explanation of the output of `lscpu`.  
		* Finally, build the julia executable with `sudo make -j N` where `N` is the number of CPUs on the cloud instance.  It took about ten minutes to build from source on a *p2.xlarge* AWS instance with 4 CPUs.
		* Add a link to the `/usr/local/bin` directory that puts the `julia` executable in the path with the link command `sudo ln -f -s ~/julia/julia /usr/local/bin/julia`.  This allows you to issue the `julia` command from anywhere and it will launch the REPL or invoke julia to run a program.
		* The build instructions also recommend running `make testall` before using the executable and this takes about ten minutes to run all the tests. 
		* Once built, launch Julia and `Pkg.add("Mocha")`.  Once Mocha loads run `Pkg.test("Mocha")` to ensure that all components in the package are working.  I've had problems here and have solved them by removing Mocha, re-adding it and once I had to ensure the git branch was set to master instead of `<HEAD>`.  After Mocha finishes all of its tests then exit the Julia REPL.
