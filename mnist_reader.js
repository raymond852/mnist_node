var fs = require('fs');

function clone(obj) {
  // Handle the 3 simple types, and null or undefined
  if (null == obj || "object" != typeof obj) return obj;

  // Handle Date
  if (obj instanceof Date) {
    var copy = new Date();
    copy.setTime(obj.getTime());
    return copy;
  }

  // Handle Array
  if (obj instanceof Array) {
    var copy = [];
    for (var i = 0, len = obj.length; i < len; i++) {
      copy[i] = clone(obj[i]);
    }
    return copy;
  }

  // Handle Object
  if (obj instanceof Object) {
    var copy = {};
    for ( var attr in obj) {
      if (obj.hasOwnProperty(attr)) copy[attr] = clone(obj[attr]);
    }
    return copy;
  }

  throw new Error("Unable to copy obj! Its type isn't supported.");
}

function vetorize(num, dim) {
  if (num >= dim) {
    throw 'number should be less than dimension';
  }
  var ret = [];
  var dimension = dim;
  while (dim) {
    if (dim + num === dimension) {
      ret.push(1);
    }
    else {
      ret.push(0);
    }
    dim--;
  }
  return ret;
}

var readTrain = function(num_train, offset) {
  var imgbuf = fs.readFileSync('train-images-idx3-ubyte');
  var labelbuf = fs.readFileSync('train-labels-idx1-ubyte');
  var imgOffset = 16;
  var labelOffset = 8;
  var training_data = [];
  var training_data_elem = {
    x : [],
    y : []
  };
  for (var i = 0; i < num_train; i++) {
    training_data_elem.x = [];
    training_data_elem.y = [];
    for (var j = 0; j < 784; j++) {
      if (offset) {
        training_data_elem.x.push(imgbuf[imgOffset + j + (i + offset) * 784]);
      }
      else {
        training_data_elem.x.push(imgbuf[imgOffset + j + i * 784]);
      }
    }
    if (offset) {
      training_data_elem.y = vetorize(labelbuf[labelOffset + i + offset], 10);
    }
    else {
      training_data_elem.y = vetorize(labelbuf[labelOffset + i], 10);
    }
    training_data.push(clone(training_data_elem));
  }
  return training_data;
};

var readTest = function(num_test) {
  var imgbuf = fs.readFileSync('t10k-images-idx3-ubyte');
  var labelbuf = fs.readFileSync('t10k-labels-idx1-ubyte');
  var imgOffset = 16;
  var labelOffset = 8;
  var test_data = [];
  var test_data_elem = {
    x : [],
    y : []
  };
  for (var i = 0; i < num_test; i++) {
    test_data_elem.x = [];
    test_data_elem.y = [];
    for (var j = 0; j < 784; j++) {
      test_data_elem.x.push(imgbuf[imgOffset + j + i * 784]);
    }
    test_data_elem.y = vetorize(labelbuf[labelOffset + i], 10);
    test_data.push(clone(test_data_elem));
  }
  return test_data;
};

var readImgWithIndex = function(index, file) {
  var buffer = new Buffer(784);
  var fd = fs.openSync(file, 'r');
  fs.readSync(fd, buffer, 0, 784, 16 + 784 * index);
  var ret = [];
  for (var i = 0; i < 784; i++) {
    ret.push(buffer[i]);
  }
  fs.closeSync(fd);
  return ret;
};

var readLabelWithIndex = function(index, file) {
  var buffer = new Buffer(16);
  var fd = fs.openSync(file, 'r');

  fs.readSync(fd, buffer, 0, 1, 8 + parseInt(index));
  fs.closeSync(fd);
  return vetorize(buffer[0], 10);
};

var readTrainImgWithIndex = function(index) {
  return readImgWithIndex(index, 'train-images-idx3-ubyte');
};

var readTestImgWithIndex = function(index) {
  return readImgWithIndex(index, 't10k-images-idx3-ubyte');
};

var readTrainLabelWithIndex = function(index) {
  return readLabelWithIndex(index, 'train-labels-idx1-ubyte');
};

var readTestLabelWithIndex = function(index) {
  return readLabelWithIndex(index, 't10k-labels-idx1-ubyte');
};

exports.readTrain = readTrain;
exports.readTest = readTest;
exports.readTrainImgWithIndex = readTrainImgWithIndex;
exports.readTestImgWithIndex = readTestImgWithIndex;
exports.readTrainLabelWithIndex = readTrainLabelWithIndex;
exports.readTestLabelWithIndex = readTestLabelWithIndex;