// import logo from './logo.svg';
import React from 'react';
import { InteractiveForceGraph, ForceGraph, ForceGraphNode, ForceGraphLink} from 'react-vis-force';
import Container from 'react-bootstrap/Container';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import DataBuilder from './Config';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

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

// var data = JSON.parse(require('./data.json'));

const NODE_KEY = "id"       // Allows D3 to correctly update DOM

class Graph extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      data: null,
      config: null,
      selected: {},
      graph: <div>{'no graph'}</div>
    }
    this.handleCallback = this.handleCallback.bind(this)
    this.componentDidUpdate = this.componentDidUpdate.bind(this)
  }

  /* Define custom graph editing methods here */
  handleCallback = (childData) =>{
    this.setState({
                  graph: null,
                  data: childData.data,
                  config: childData.config})
  }

  componentDidUpdate () {
    if(this.state.data){
      var nodes = this.state.data.nodes;
      var edges = this.state.data.edges;
      var selected = this.state.selected;
  
      var NodeTypes = this.state.config.NodeTypes;
      var NodeSubtypes = this.state.config.NodeSubtypes;
      var EdgeTypes = this.state.config.EdgeTypes;
      var graph = <GraphView  
                    // ref='GraphView'
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
                    
      this.setState({data:null,
                    graph: graph})
    } 
  }

  render() {
    return (
      <Container id='graph' style={{height: '1000px'}}>
        <Row>
          <Col>
          <DataBuilder parentCallback = {this.handleCallback} />
          </Col>
          <Col>
          {this.state.graph}
          </Col>
        </Row>
      </Container>
    );
  }

}

export default Graph;
