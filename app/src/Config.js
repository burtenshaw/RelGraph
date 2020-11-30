import React from 'react';
import { Button, ButtonGroup, Container, Dropdown, InputGroup, FormControl, Form, ListGroup, ListGroupItem, OverlayTrigger, Popover, Tab, Tabs } from 'react-bootstrap';
import Highlighter from "react-highlight-words";

class RelationView extends React.Component {

    constructor(props) {
        super(props);
    }

    // onSelectPanNode = (event) => {
    //     if (this.GraphView) {
    //       this.ref.GraphView.panToNode(event.target.value, true);
    //     }
    //   };

    render () {

        var relation = this.props.relations;

        return (

                <Tabs defaultActiveKey="info" >
                <Tab eventKey="info" title="Relation Info">

                    <ListGroup>
                        <ListGroupItem>
                            <ButtonGroup>
                                <Button variant = 'outline-secondary' >source</Button>
                                <Button variant = 'outline-secondary' >{relation.top_sentence}</Button>
                                <Button variant = 'outline-secondary' >target</Button>
                            </ButtonGroup>
                        </ListGroupItem>
                        <ListGroupItem>
                        <ButtonGroup>
                                <Button variant = 'outline-secondary' >size</Button>
                                <Button variant = 'outline-secondary' >{relation.size}</Button>
                            </ButtonGroup>
                        </ListGroupItem>
                        <ListGroupItem>

                            {relation.key_words.map(kw => {
                                        return (<Button variant = 'outline-secondary' style={{margin : '2px'}}>{kw}</Button>)
                            }
                                
                                )}
                        </ListGroupItem>
                    </ListGroup>

                </Tab>
                <Tab eventKey="profile" title="Examples">
                    <ListGroup>
                        {relation.relations.map( r => {
                            return (
                                <ListGroupItem>
                                    <ButtonGroup className="d-flex">
                                        <Button variant = 'outline-secondary'>{r.s}</Button>


                                        <OverlayTrigger
                                            trigger="click"
                                            key='paragraph'
                                            placement='top'
                                            overlay={
                                                <Popover id={`popover-positioned-top`}>
                                                <Popover.Title as="h3">book placement</Popover.Title>
                                                <Popover.Content>
                                                <Highlighter
                                                        highlightClassName="highlighted"
                                                        searchWords={[r.rc]}
                                                        autoEscape={true}
                                                        textToHighlight={r.p}
                                                    />
                                                </Popover.Content>
                                                </Popover>
                                            }
                                            >
                                            <Button block variant = 'outline-secondary'>{r.rc.slice(0,20)}...</Button>
                                            </OverlayTrigger>
                                            
                                            <Button variant = 'outline-secondary'>{r.t}</Button>
                                        {/* add overlay */}
                                    </ButtonGroup>
                                </ListGroupItem>
                            )
                    })}     
                    </ListGroup>            
                </Tab>

                </Tabs>
            
        )
    }


}

class DataBuilder extends React.Component {

    constructor(props) {
        super(props);

        var config = JSON.parse(require('./config.json'));
        var data = JSON.parse(require('./data.json'));
        var relationslist = JSON.parse(require('./relations.json'));
        
        this.state = {
            config : this.buildconfig(config),
            data : this.builddata(data),
            query : [],
            newQuery : null,
            newData : true,
            relations : config.EdgeTypes,
            relationview : null,
            relationid : null,
            relationslist :relationslist
        };

        this.buttonClick = this.buttonClick.bind(this);
        this.handleClusterQuery = this.handleClusterQuery.bind(this);
        this.getRelation = this.getRelation.bind(this);
        this.onSelectRelation = this.onSelectRelation.bind(this);
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

        const response = await fetch('http://localhost:5000/cluster', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(this.state.query)
        })

        const data = await response.json()

        console.log(data)

        this.setState({
            data: this.builddata(data.data),
            config: this.buildconfig(data.config)})

      }
    

    buttonClick = (event) => {

        if (this.state.newQuery) {
            // this.props.parentCallback({data: null,
            //     config: null});
            // this.setState({ newQuery : false });
            // this.getRelation();
        }
        
        // this.props.parentCallback({data: this.state.data,
        //                             config: this.state.config});

        // event.preventDefault();
    }


    handleClusterQuery = (event) => {
        this.state.query.push(event);
        this.setState({ newQuery : true});
        this.props.parentCallback({query: this.state.query});
        // event.preventDefault();
    }


    onSelectRelation = (e) => {
        this.setState({relationid : parseInt(e.cluster)})   
        
    }

    // componentDidMount () {
    //     this.props.parentCallback({data: this.state.data,
    //         config: this.state.config});
    // }

    render () {

        var relationview = (this.state.relationid) ? <RelationView ref = {this.props.ref} relations = {this.state.relationslist[this.state.relationid]} /> : <div></div>

        return (

            <Container>
                <Form.Group style = {{ height : "30em", overflow : "scroll"}}>
                    {this.state.relations.map(relation => {
                        return (
                            
                            <Form.Row  >
                            <InputGroup>
                            <InputGroup.Prepend>
                                <InputGroup.Checkbox onClick = {(event) => this.handleClusterQuery(relation.typeText)} />
                            </InputGroup.Prepend>
                                
                                    <InputGroup.Prepend>
                                    <InputGroup.Text id="basic-addon1" 
                                                        style = {{backgroundColor : relation.color, width : '100px', color : "white"}} 
                                                        onClick = {(event) => this.onSelectRelation(relation)}>{relation.typeText}</InputGroup.Text>
                                    </InputGroup.Prepend>
                                    <FormControl
                                    placeholder={'rename'}
                                    />
                                
                            </InputGroup>

                            </Form.Row>
                            
                                )
                    })}
                    </Form.Group>
                <Container style = {{ height : '30em', overflow : 'scroll'}}>
                    {relationview}
                </Container>
                <ButtonGroup>
                    <Button onClick = {(e) => this.buttonClick(e)}> Get Data </Button>
                </ButtonGroup>
                
            </Container>
            
        )}
}


export default DataBuilder;