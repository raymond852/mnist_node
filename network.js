var numeric = require('numeric');
var reader = require('./mnist_reader');

var input_scale = 255.0;

function rnd_snd() {
  return (Math.random() * 2 - 1) + (Math.random() * 2 - 1)
      + (Math.random() * 2 - 1);
}

function rnd(mean, stdev) {
  return Math.round(rnd_snd() * stdev + mean);
}

function gen_norm_rnd_martix(s) {
  var ret_rnd_fun = function(s, k) {
    var i, n = s[k], ret = Array(n), rnd;
    if (k === s.length - 1) {
      rnd = rnd_snd;
      for (i = n - 1; i >= 1; i -= 2) {
        ret[i] = rnd();
        ret[i - 1] = rnd();
      }
      if (i === 0) {
        ret[0] = rnd();
      }
      return ret;
    }
    for (i = n - 1; i >= 0; i--) {
      ret[i] = ret_rnd_fun(s, k + 1);
    }
    return ret;
  };
  return ret_rnd_fun(s, 0);
}

function gen_zero_martix(s) {
  var ret_rnd_fun = function(s, k) {
    var i, n = s[k], ret = Array(n);
    if (k === s.length - 1) {
      for (i = n - 1; i >= 1; i -= 2) {
        ret[i] = 0;
        ret[i - 1] = 0;
      }
      if (i === 0) {
        ret[0] = 0;
      }
      return ret;
    }
    for (i = n - 1; i >= 0; i--) {
      ret[i] = ret_rnd_fun(s, k + 1);
    }
    return ret;
  };
  return ret_rnd_fun(s, 0);
}

function assert(condition, message, callback) {
  if (!condition) {
    if (callback) callback();
    throw message || "Assertion failed";
  }
}

function shuffle(array) {
  for (var i = array.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
  return array;
}

function scale(f, m) {
  var ret = [];
  for (var i = 0; i < m.length; i++) {
    if (m[i] instanceof Array) {
      ret.push(scale(f, m[i]));
    }
    else {
      ret.push(m[i] * f);
    }
  }
  return ret;
}

function flipsign(z) {
  return scale(-1, z);
}

function hproduct(a, b) {
  assert(a.length === b.length,
      'hproduct:the length of each parameter should be the same');
  var ret = [];
  for (var i = 0; i < a.length; i++) {
    ret.push(a[i] * b[i]);
  }
  return ret;
}

var Network = function(layers) {
  assert(layers instanceof Array, 'layers should be an array');
  var num_layers = layers.length;
  var sizes = layers;
  var biases = [];

  this.init_bias = function() {
    layers.slice(1).forEach(function(elem, index, array) {
      biases.push(gen_norm_rnd_martix([elem
      ]));
    });
  };

  var weights = [];
  this.init_weight = function() {
    var rows = layers.slice(1);
    var cols = layers.slice(0, -1);
    assert(rows.length === cols.length,
        'rows length should equals to cols length');
    for (var i = 0; i < rows.length; i++) {
      weights.push(gen_norm_rnd_martix([rows[i],cols[i]
      ]));
    }
  };

  var optimal_weights;
  var optimal_biases;
  var drop_count = 2;

  this.init = function() {
    this.init_bias();
    this.init_weight();
  };

  this.init();

  this.initWithParas = function(w, b) {
    weights = w;
    biases = b;
  };

  this.feedforward = function(input) {
    assert(biases.length === weights.length,
        'biases.length should equals to weights.length');
    for (var i = 0; i < biases.length; i++) {
      var w = weights[i];
      var b = biases[i];
      input = this.sigmoid_vec(numeric.add(numeric.dot(w, input), b));
    }
    return input;
  };

  this.SGD = function(training_data, epochs, mini_batch_size, eta, valid_data) {
    var len_test;
    if (valid_data) {
      len_test = valid_data.length;
    }
    for (var i = 0; i < epochs; i++) {
      var d_count = 0;
      var last_accuracy;
      shuffle(training_data);
      var mini_batches = [];
      for (var j = 0; j < training_data.length; j += mini_batch_size) {
        mini_batches.push(training_data.slice(j, j + mini_batch_size));
      }
      for (var k = 0; k < mini_batches.length; k++) {
        this.update_mini_batch(mini_batches[k], eta);
      }
      if (valid_data) {
        var curr_accuracy = this.evaluate(valid_data);
        if (last_accuracy && curr_accuracy >= last_accuracy) {
          optimal_weights = weights;
          optimal_biases = biases;
          d_count = 0;
        }
        else if (last_accuracy && curr_accuracy < last_accuracy) {
          d_count++;
        }
        last_accuracy = curr_accuracy;
        console.log("Epoch: %d: %d/%d", i, curr_accuracy, len_test);
        if (d_count >= drop_count) {
          break;
        }
      }
      else {
        console.log("Epoch %d % complete", i / epochs);
      }
    }
    if (optimal_weights && optimal_biases) {
      this.initWithParas(optimal_weights, optimal_biases);
    }
  };

  this.update_mini_batch = function(mini_batch, eta) {
    var nabla_b = [];
    var nabla_w = [];
    assert(biases.length === weights.length,
        'biases.length should be equal to weights.length');
    for (var i = 0; i < biases.length; i++) {
      var nabla_b_push = gen_zero_martix(numeric.dim(biases[i]));
      nabla_b.push(nabla_b_push);
      var nabla_w_push = gen_zero_martix(numeric.dim(weights[i]));
      nabla_w.push(nabla_w_push);
    }

    assert(nabla_b.length === nabla_w.length,
        'nabla_w.length should be equal to nabla_b.length');
    for (var j = 0; j < mini_batch.length; j++) {
      var ret_backprop = this.backprop(mini_batch[j].x, mini_batch[j].y);
      var delta_nabla_b = ret_backprop.x;
      var delta_nabla_w = ret_backprop.y;
      for (var k = 0; k < nabla_b.length; k++) {
        nabla_b[k] = numeric.add(nabla_b[k], delta_nabla_b[k]);
        nabla_w[k] = numeric.add(nabla_w[k], delta_nabla_w[k]);
      }

      assert(weights.length === biases.length,
          'weights.length should be equal to biases.length');
      assert(nabla_b.length === biases.length,
          'nabla_b.length should be equal to biases.length');
      for (var l = 0; l < weights.length; l++) {

        weights[l] = numeric.add(weights[l], scale(-eta, nabla_w[l]));
        biases[l] = numeric.add(biases[l], scale(-eta, nabla_b[l]));
      }
    }

  };

  this.backprop = function(x, y) {
    var nabla_b = [];
    var nabla_w = [];
    assert(biases.length === weights.length,
        'biases.length should be equal to weights.length');
    for (var i = 0; i < biases.length; i++) {
      var nabla_b_push = gen_zero_martix(numeric.dim(biases[i]));
      nabla_b.push(nabla_b_push);
      var nabla_w_push = gen_zero_martix(numeric.dim(weights[i]));
      nabla_w.push(nabla_w_push);
    }

    // feedforward
    var activation = scale(1 / input_scale, x);
    var activations = [scale(1 / input_scale, x)
    ];

    var zs = [];
    assert(biases.length === weights.length,
        'biases.length should be equal to weights.length');
    var z;
    for (var j = 0; j < biases.length; j++) {

      z = numeric.add(numeric.dot(weights[j], activation), biases[j]);

      zs.push(z);
      activation = this.sigmoid_vec(z);
      activations.push(activation);
    }

    // backward process
    var delta = hproduct(this.cost_derivative(
        activations[activations.length - 1], y), this
        .sigmoid_prime_vec(zs[zs.length - 1]));
    nabla_b[nabla_b.length - 1] = delta;

    var last_2_activation_matrix = [activations[activations.length - 2]
    ];

    nabla_w[nabla_w.length - 1] = numeric.dot(numeric.transpose([delta
    ]), last_2_activation_matrix);

    for (var l = 2; l < num_layers; l++) {
      z = zs[zs.length - l];
      var spv = this.sigmoid_prime_vec(z);
      delta = hproduct(numeric.transpose(numeric.dot(numeric
          .transpose(weights[weights.length - l + 1]), numeric.transpose([delta
      ])))[0], spv);

      nabla_b[nabla_b.length - l] = delta;
      var last_lplus1_activation_matrix = [activations[activations.length - l
          - 1]
      ];

      nabla_w[nabla_w.length - l] = numeric.dot(numeric.transpose([delta
      ]), last_lplus1_activation_matrix);

    }
    return {
      x : nabla_b,
      y : nabla_w
    };
  };

  this.cost_derivative = function(output_activation, y) {
    return numeric.add(output_activation, flipsign(y));
  };

  this.evaluate = function(valid_data) {
    var rights = 0;

    for (var i = 0; i < valid_data.length; i++) {
      var network_out = this.feedforward((valid_data[i]).x);
      var eval_index = network_out.indexOf(Math.max.apply(Math, network_out));
      var true_index = valid_data[i].y.indexOf(1);
      if (eval_index === true_index) {
        rights++;
      }
    }
    return rights;
  };

  this.sigmoid = function(z) {
    return 1.0 / (1.0 + Math.exp(-z));
  };

  this.sigmoid_vec = function(v) {
    var ret = [];
    assert(v instanceof Array,
        'parameter of sigmoid_vec function should be an array');
    for (var i = 0; i < v.length; i++) {
      ret.push(this.sigmoid(v[i]));
    }
    return ret;
  };

  this.sigmoid_prime = function(z) {
    return this.sigmoid(z) * (1 - this.sigmoid(z));
  };

  this.sigmoid_prime_vec = function(v) {
    var ret = [];
    assert(v instanceof Array,
        'parameter of sigmoid_prime_vec function should be an array');
    for (var i = 0; i < v.length; i++) {
      ret.push(this.sigmoid_prime(v[i]));
    }
    return ret;
  };

  this.getWeights = function() {
    return weights;
  };

  this.getBiases = function() {
    return biases;
  };
};

exports.Network = Network;
