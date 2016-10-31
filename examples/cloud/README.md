# Mocha in the Cloud

The real power of developing deep learning networks is only realized
when you can test your ideas and run your models on powerful compute
platforms that complete training in a fraction of the time it took only
a few years ago. Today, state-of-the-art machine learning algorithms
routinely run on cloud based compute that offers access to cutting edge
GPU technology for a tiny cost compared to buying the GPU hardware and
running it in your personal or company servers.

Amazon Web Services (AWS) is one of the most popular cloud based
services that provide access to such powerful computers. As Mocha and
Julia mature I'm sure that a pre-configured Amazon Machine Image (AMI)
will emerge to run Mocha in the cloud. Now, in October 2016, such an AMI
does not exist, but even when a pre-configured image for Mocha does
become available I highly recommend following through this tutorial at
least once so you understand how cloud resources are provisioned and
configured to suport Deep Learning.

We are going to show you how to take the CIFAR-10 example and get it
running in the cloud. Along the way you will learn how to interact with
AWS and get a broader understanding of cloud architectures in general.

The example includes several images and code snapshots.  So it is best
to follow along with the [documentation](http://mochajl.readthedocs.io/en/latest/tutorial/cloud.html). 