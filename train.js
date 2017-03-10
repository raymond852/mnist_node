var reader = require('./mnist_reader');
var io = require('socket.io-client');

var Network = require('./network').Network;

var data = reader.readTrain(50000);
var valid_data = reader.readTrain(10000, 50000);

var net = new Network([784,32,10
]);
// net.SGD(data, 10, 10, 0.03, valid_data);
var socket = io.connect('http://localhost:3333');
socket.on('connect', function() {

  console.log('connected to server');
  
  socket.emit('updateNetwork', {
    w : net.getWeights(),
    b : net.getBiases()
  });


});

socket.on('disconnect', function() {
  console.log('disconneted');
});


