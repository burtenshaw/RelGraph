// import logo from './logo.svg';
import React from 'react';
import { unmountComponentAtNode } from 'react-dom';
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
import { Button } from 'react-bootstrap';

// var data = JSON.parse(require('./data.json'));

const NODE_KEY = "id"       // Allows D3 to correctly update DOM

class Graph extends React.Component {

  constructor(props) {
    super(props);
    var config = JSON.parse(require('./config.json'));
    var data = JSON.parse(require('./data.json'));
    this.state = {
      graph: data,
      config: this.buildconfig(config),
      selected: {}
    }
    this.handleCallback = this.handleCallback.bind(this)
    this.componentDidUpdate = this.componentDidUpdate.bind(this)
    // this.onSelectPanNode = this.onSelectPanNode.bind(this)
    // this.onSelectEdge = this.onSelectEdge.bind(this)
    // this.componentDidMount = this.componentDidMount.bind(this)

    this.GraphView = React.createRef();

  }

  componentDidUpdate() {
    if (this.state.newQuery) {
      // this.setState({ data : {nodes : null, selected : null, edge : null}})
      this.setState({newQuery : false});
      
      var graph = this.state.graph;

      var newEdges = this.state.graph.edges.filter(e => !this.state.query.includes(e.type));
      console.log(newEdges.length);
      var deleteEdgeIds = this.state.graph.edges.filter(e => this.state.query.includes(e.type))
      console.log(deleteEdgeIds.length);

      deleteEdgeIds.map( edge => {
        this.GraphView.deleteEdgeBySourceTarget(edge.source, edge.target)

      })
    }
  }

  /* Define custom graph editing metho  ds here */
  handleCallback = (childData) =>{
    this.setState({query: childData.query, newQuery : true});

  }

  buildconfig(config) {
        
    var NodeTypes = {}

    config.NodeTypes.map( type => {
        type.shape =  (
                <symbol viewBox="0 0 100 100" id={type.typeText} key="0">
                    <circle cx="50" cy="50" r="45" style={{ color : '#ffffff' , 
                                                            fill : type.color}} ></circle>
                </symbol>
                )
        NodeTypes[type.typeText] = type
        
    })
    
    var EdgeTypes = {}
    
    config.EdgeTypes.map( type => {
        type.shape = (
                <symbol viewBox="0 0 200 200" id={type.typeText} key="0" label_from = {type.typeText}>
                    <circle cx="100" cy="100" r="45" fill={type.color}></circle>
                </symbol>
        )
        EdgeTypes[type.typeText] = type
    })
    
    var PotterConfig =  {
        NodeTypes: NodeTypes,
        NodeSubtypes: {},
        EdgeTypes: EdgeTypes
      }
      
    return PotterConfig;
  }

  render() {

    return (
      <Container id='graph' style={{height: '100%'}}>
        <Row>
          <Col md={6}>
            <DataBuilder ref={this.GraphView} parentCallback = {this.handleCallback} />
          </Col>
          <Col md={6}>
            <GraphView  
              ref={el => (this.GraphView = el)}
              nodeKey={NODE_KEY}
              nodes={this.state.graph.nodes}
              edges={this.state.graph.edges}
              selected={this.state.selected}
              nodeTypes={this.state.config.NodeTypes}
              nodeSubtypes={this.state.config.NodeSubtypes}
              edgeTypes={this.state.config.EdgeTypes}
              onSelectNode={this.onSelectNode}
              onCreateNode={this.onCreateNode}
              onUpdateNode={this.onUpdateNode}
              onDeleteNode={this.onDeleteNode}
              onSelectEdge={this.onSelectEdge}
              onCreateEdge={this.onCreateEdge}  
              onSwapEdge={this.onSwapEdge}
              onDeleteEdge={this.onDeleteEdge}
              layoutEngineType={'SnapToGrid'}/>
          </Col>
        </Row>
        
      </Container>
    );
  }

}



export default Graph;
