import React from 'react';
import { Button, ButtonGroup, Container, Dropdown, Form, ListGroup, ListGroupItem } from 'react-bootstrap';


class DataBuilder extends React.Component {

    constructor(props) {
        super(props);

        var config = JSON.parse(require('./config.json'));
        var data = JSON.parse(require('./data.json')) 

        this.state = {
            config : this.buildconfig(config),
            data : this.builddata(data),
            query : [],
            newQuery : null,
            newData : true,
            relations : config.EdgeTypes,
            relationview : 'none'
        };

        this.buttonClick = this.buttonClick.bind(this);
        this.handleClusterQuery = this.handleClusterQuery.bind(this);
        this.getRelation = this.getRelation.bind(this);
        // this.doData = this.doData.bind(this);

        // this.componentDidUpdate = this.componentDidUpdate.bind(this);
        // this.componentDidMount = this.componentDidUpdate.bind(this);
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

    builddata(data) {  
        return data
    }

    async getRelation() {

        const response = await fetch('http://localhost:5001/cluster', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(this.state.query)
        })

        const data = await response.json()

        this.setState({
            data: this.builddata(data.data),
            config: this.buildconfig(data.config)})

      }
    

    buttonClick = (event) => {

        if (this.state.newQuery) {
            this.props.parentCallback({data: null,
                config: null});
            this.setState({ newQuery : false });
            this.getRelation();

        }
        
        this.props.parentCallback({data: this.state.data,
                                    config: this.state.config});

        event.preventDefault();
    }

    handleClusterQuery = (e) => {
        this.state.query.push(e);
        this.setState({ newQuery : true});
    }

    // componentDidUpdate() {

    // }

    // componentDidMount () {
    //     // this.doData();
    // }

    onSelectRelation = (e) => {
        this.setState({relationview : e.typeText})
    } 

    render () {
        return (

            <Container>
                <ListGroup style = {{ height : "30em", overflow : "scroll"}}>
                    {this.state.relations.map(relation => {
                        return (
                        
                        <ListGroupItem onClick = {(event) => this.onSelectRelation(relation)} style = {{backgroundColor : relation.color, color : "white"}}>

                                <Form.Group>
                                    <Form.Row>
                                    <Form.Check inline 
                                                type="checkbox" 
                                                style={{float: 'right'}}
                                                onClick = {(event) => this.handleClusterQuery(relation.typeText)} />
                                    
                                    <Form.Label>
                                        {relation.typeText}
                                    </Form.Label>
                                    </Form.Row>
                                </Form.Group>
                                </ListGroupItem>)
                    })}
                </ListGroup>
                <Container style = {{ height : '30em', overflow : 'hidden'}}>
                    {this.state.relationview}
                </Container>
                <Button onClick = {(e) => this.buttonClick(e)}> Get Data </Button>
            </Container>

            // <ButtonGroup>
            //     <Dropdown>
            //         <Dropdown.Toggle variant="success" id="dropdown-basic">
            //             Dropdown Button
            //         </Dropdown.Toggle>

            //         <Dropdown.Menu >
            //             <Dropdown.Item onSelect = {this.handleClusterQuery} eventKey = 'cluster0' >cluster0</Dropdown.Item>
            //             <Dropdown.Item onSelect = {this.handleClusterQuery} eventKey = 'cluster1' >cluster1</Dropdown.Item>
            //             <Dropdown.Item onSelect = {this.handleClusterQuery} eventKey = 'cluster2' >cluster2</Dropdown.Item>
            //         </Dropdown.Menu>
            //     </Dropdown>
                
            // </ButtonGroup>
            
        )}
}


export default DataBuilder;