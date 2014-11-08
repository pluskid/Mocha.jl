function test_mocha_kernels(sys::System)
  println("-- Testing Mocha CUDA kernels")

  input_blob = make_blob(sys.backend, Float32, 1)
  copy!(input_blob, Float32[0.1])
  n = 3840
  output_blob = make_blob(sys.backend, Float32, 1)
  copy!(output_blob, Float32[0])
  x_block = int(ceil(float64(n)/CUDA.THREADS_PER_BLOCK))
  CUDA.launch(sys.backend.mocha.test, x_block, CUDA.THREADS_PER_BLOCK,
      (n, input_blob.ptr.p, output_blob.ptr.p))
  output = Float32[0]
  copy!(output, output_blob)
  output = output[1]
  
  real_output = 0f0
  for i = 1:n
    real_output += -log(0.1f0)
  end
  println("output = $output")
  println("real_output = $real_output")


  println("    > logistic loss forward")
  for data_type in [Float32, Float64]
    if data_type == Float32
      eps = 1e-7
    else
      eps = 1e-2
    end

    h, w, c, n = (1, 1, 1, 3840)
    prob = abs(rand(data_type, (h,w,c,n))) + 0.1
    label = abs(rand(Int, (h, w, 1, n))) % c
    label = convert(Array{data_type}, label)

    prob_blob = make_blob(sys.backend, data_type, size(prob))
    copy!(prob_blob, prob)
    label_blob = make_blob(sys.backend, data_type, size(label))
    copy!(label_blob, label)

    # This should always be float32
    loss_blob = make_blob(sys.backend, Float32, 1)
    copy!(loss_blob, Float32[0])

    spatial_dim = h*w
    prob_dim = c

    x_block = int(ceil(float64(n)/CUDA.THREADS_PER_BLOCK))
    y_block = spatial_dim

    if data_type == Float32
      kernel = sys.backend.mocha.logistic_loss_forward_float
    else
      kernel = sys.backend.mocha.logistic_loss_forward_double
    end
    CUDA.launch(kernel, (x_block, y_block), (CUDA.THREADS_PER_BLOCK, 1),
        (prob_blob.ptr.p, label_blob.ptr.p, n, spatial_dim, prob_dim, loss_blob.ptr.p))

    loss = Float32[0]
    copy!(loss, loss_blob)
    loss = loss[1]

    expected_loss = 0f0
    for i = 1:n
      for j = 1:w
        for k = 1:h
          expected_loss += -log(prob[k, j, int(label[k,j,1,i])+1, i])
        end
      end
    end

    println("loss = $loss")
    println("expected_loss = $expected_loss")
    @test (-eps < expected_loss - loss < eps)
  end
end

test_mocha_kernels(sys_cudnn)
