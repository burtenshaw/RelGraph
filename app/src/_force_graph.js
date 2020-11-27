// import logo from './logo.svg';
import React from 'react';
import { InteractiveForceGraph, ForceGraph, ForceGraphNode, ForceGraphLink} from 'react-vis-force';
import Container from 'react-bootstrap/Container';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';


var data = JSON.parse(require('./data.json'));


// const NODE_KEY = "id"       // Allows D3 to correctly update DOM

class Graph extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      graph: data,
      selected: {}
    }
  }



  render() {
    const nodes = this.state.graph.nodes;
    const edges = this.state.graph.edges;
    return (
      <Container>
      <InteractiveForceGraph
            simulationOptions={{ height: 800, width: 800 }}
            labelAttr="label"
            highlightDependencies
            // onSelectNode={(node) => console.log(node)}
          >
      {nodes.map(n => {
        console.log(n)
        return ( <ForceGraphNode label = {n.title} node={{ id: n.id }} fill={n.color} className={n.class}/>)
      })
    }
        {edges.map(e => {
        return (<ForceGraphLink link={{ source: e.source, target: e.target }} stroke={e.color} className={e.class}/>) 
      })}

      </InteractiveForceGraph>
      </Container>
    );
  }

}

export default Graph;
