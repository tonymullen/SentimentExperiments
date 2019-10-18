var fs = require('fs');
var ip = require('ip');
var qs = require('querystring');
var http = require('http');
var request = require('request');
var argparse = require('argparse');

var parser = new argparse.ArgumentParser();
parser.addArgument([ '-p', '--port' ]);
parser.addArgument([ '-a', '--anserve' ]);
var args = parser.parseArgs();
var PORT = args.port ? args.port : 8081;
var ANNOT_SERVER = args.anserve ? args.anserve : 'http://localhost:3030'

var server = http.createServer(function(req, res) {
  if ('/' == req.url) {
    console.log("Serving webpage");
    fs.readFile('annotator.html', function (err, data) {
      res.writeHead( 200, {
        'content-type': 'text/html',
        'Access-Control-Allow-Origin': 'http://localhost:' + PORT
      } );
      // Make sure the client script knows what port we're on
      var datastring = data.toString()
                          .replace('PORT_DUMMY', PORT)
                          .replace('ADDRESS_DUMMY', ip.address());
      res.end(datastring);
    });
  } else if ('/api' == req.url) {
    console.log('Getting sentence');
    switch (req.method) {
      case 'GET':
        console.log("Requesting new sentence from: " + ANNOT_SERVER);
        request(ANNOT_SERVER, (err, response, body) => {
          res.writeHead(200, {
            'Access-Control-Allow-Origin': 'http://localhost:' + PORT
          });
          res.end(body);
        });
        break;
      case 'POST':
      var body = '';
        req.on('data', function (data) {
          body += data;
        });
        req.on('end', function () {
          var post = qs.parse(body);
          request.post(ANNOT_SERVER, {form: post});
          res.writeHead(200, {
            'Content-Type': 'text/html',
            'Access-Control-Allow-Origin': 'http://localhost:' + PORT
          });
          res.end('post received');
        });
        console.log("Posting judgment");
        break;
      }
  }
});

server.listen(PORT, function() {
  console.log("Application server listening on port " + PORT);
});
