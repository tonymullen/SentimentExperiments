To run the server you need to install node.js.

In the project directory, run

`npm install`

Start the dummy annotation server with:

`node dummy_annotation_server.js`

Start the application server with:

`node server.js`

The server defaults to localhost:8081. You can set the port with the `-p` command line option. You can use the `-a` option to tell the server where the annotation server is running. The default is localhost:3030.

To do annotations, you must input your username. The acceptable usernames are 'alice', 'bob', and 'charlie'. Any of those will do.
