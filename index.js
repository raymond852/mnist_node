var express = require('express');
var fs = require('fs');
var reader = require('./mnist_reader');
var app = express();
var http = require('http'), server = http.createServer(app), io = require(
    'socket.io').listen(server);
var Network = require('./network').Network;


var net = new Network([784,32,10
]);

io.sockets.on('connection', function(socket) {
  socket.on('updateNetwork', function(data) {
	  console.log(data);
    net.initWithParas(data.w, data.b);
  });
});

// render home page
app.get(/^\/(index.html)?$/, function(req, res) {
  var homepage = fs.readFileSync('index.html');
  res.writeHead(200, {
    'Content-Type' : 'text/html',
    'Content-Length' : homepage.length
  });
  res.write(homepage);
  res.end();
});

app.get("/select_index", function(req, res) {
  var index = req.query.index;

  var test_data = {};
  test_data.y = reader.readTestLabelWithIndex(index);
  test_data.x = reader.readTestImgWithIndex(index);

  var network_out = net.feedforward(test_data.x);

  var data = {};
  data.imgData = test_data.x;
  data.networkData = network_out;
  data.labelData = test_data.y;
  data.predictData = network_out.indexOf(Math.max.apply(Math, network_out));

  var datastr = JSON.stringify(data);
  res.writeHead(200, {
    'Content-Type' : 'application/json',
    'Content-Length' : datastr.length
  });
  res.write(datastr);
  res.end();
});

app.get('/test_all', function(req, res) {
  var ret = [];
  for (var i = 0; i < 10000; i++) {
    var network_out = net.feedforward(reader.readTestImgWithIndex(i));
    var labelIndex = reader.readTestLabelWithIndex(i).indexOf(1);
    var predictIndex = network_out.indexOf(Math.max.apply(Math, network_out));
    if (labelIndex !== predictIndex) {
      ret.push(i);
    }
  }

  var datastr = ret.toString();
  res.writeHead(200, {
    'Content-Type' : 'text/plain',
    'Content-Length' : datastr.length
  });
  res.write(datastr);
  res.end();
});

server.listen(3333);
console.log('listening on port 3333');
