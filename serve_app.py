
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/cluster', methods=['GET','POST'])
def home():
    clusters = request.json
    packet = write_data(nodes, edges, node_map, edge_map, clusters)
    packet = jsonify(packet)
    return packet

CORS(app)

app.run(host='localhost', port=5000)



# %%
