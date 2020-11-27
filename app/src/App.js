// import logo from './logo.svg';
import React from 'react';
import { InteractiveForceGraph, ForceGraph, ForceGraphNode, ForceGraphLink} from 'react-vis-force';
import Container from 'react-bootstrap/Container';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import PotterConfig from './Config';

import {
  GraphView, // required
  Edge, // optional
  type IEdge, // optional
  Node, // optional
  type INode, // optional
  type LayoutEngineType, // required to change the layoutEngineType, otherwise optional
  BwdlTransformer, // optional, Example JSON transformer
  GraphUtils // optional, useful utility functions
} from 'react-digraph';


var data = JSON.parse(require('./data.json'));

const NODE_KEY = "id"       // Allows D3 to correctly update DOM

class Graph extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      graph: data,
      selected: {}
    }
  }

  /* Define custom graph editing methods here */

  render() {
    const nodes = this.state.graph.nodes;
    const edges = this.state.graph.edges;
    const selected = this.state.selected;

    const NodeTypes = PotterConfig.NodeTypes;
    const NodeSubtypes = PotterConfig.NodeSubtypes;
    const EdgeTypes = PotterConfig.EdgeTypes;

    return (
      <Container id='graph' style={{height: '1000px'}}>

        <GraphView  ref='GraphView'
                    nodeKey={NODE_KEY}
                    nodes={nodes}
                    edges={edges}
                    selected={selected}
                    nodeTypes={NodeTypes}
                    nodeSubtypes={NodeSubtypes}
                    edgeTypes={EdgeTypes}
                    onSelectNode={this.onSelectNode}
                    onCreateNode={this.onCreateNode}
                    onUpdateNode={this.onUpdateNode}
                    onDeleteNode={this.onDeleteNode}
                    onSelectEdge={this.onSelectEdge}
                    onCreateEdge={this.onCreateEdge}
                    onSwapEdge={this.onSwapEdge}
                    onDeleteEdge={this.onDeleteEdge}/>
      </Container>
    );
  }

}

export default Graph;
