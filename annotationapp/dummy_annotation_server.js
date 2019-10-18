/*
This just sends a random item from the items array on a
GET request, and handles a POST request of a correctly
annotated item.
*/
var PORT = 3030;

var items = [
{
  "sentence": "The red dog jumped over high fences and streams.",
  "warning": "This sentence is ambigous. Does \'high\' refer to both fences and streams, or only fences?",
  "wTregex": "(NP (JJ \'tregex\' NN \'string\'))",
  "parse": "(NP (JJ \'parsed\' NN \'sentence\'))"
},
{
  "sentence": "The green sheep jumped over high fences and streams.",
  "warning": "This sentence is ambigous. Does \'high\' refer to both fences and streams, or only fences?",
  "wTregex": "(NP (JJ \'tregex\' NN \'string\'))",
  "parse": "(NP (JJ \'parsed\' NN \'sentence\'))"
},
{
  "sentence": "The black fish jumped over high fences and streams.",
  "warning": "This sentence is ambigous. Does \'high\' refer to both fences and streams, or only fences?",
  "wTregex": "(NP (JJ \'tregex\' NN \'string\'))",
  "parse": "(NP (JJ \'parsed\' NN \'sentence\'))"
},
{
  "sentence": "The orange gorilla jumped over high fences and streams.",
  "warning": "This sentence is ambigous. Does \'high\' refer to both fences and streams, or only fences?",
  "wTregex": "(NP (JJ \'tregex\' NN \'string\'))",
  "parse": "(NP (JJ \'parsed\' NN \'sentence\'))"
},
{
  "sentence": "The purple hyena jumped over high fences and streams.",
  "warning": "This sentence is ambigous. Does \'high\' refer to both fences and streams, or only fences?",
  "wTregex": "(NP (JJ \'tregex\' NN \'string\'))",
  "parse": "(NP (JJ \'parsed\' NN \'sentence\'))"
}];

var http = require('http');
var qs = require('querystring');

var server = http.createServer(function(req, res) {
  if (req.method == 'GET') {
    console.log('Got request for sentence')
    var item = items[Math.floor(Math.random() * items.length)];
    res.end(JSON.stringify(item));
  } else if (req.method == 'POST') {
    var body = '';
      req.on('data', function (data) {
        body += data;
      });
      req.on('end', function () {
        var post = qs.parse(body);
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.end('post received');
        console.log("Saving annotated object to database");
        // Save the item to the DB
      });
  }
});

server.listen(PORT, function() {
  console.log("Annotation server listening on port " + PORT);
})
