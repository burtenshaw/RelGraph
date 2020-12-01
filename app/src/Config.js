import React from 'react';
import { Button, ButtonGroup, Container, Col, Dropdown, InputGroup, FormControl, Form, OverlayTrigger, Popover, Tab, Tabs, Row, Nav } from 'react-bootstrap';
import Highlighter from "react-highlight-words";


const book_info = {
    b1: "Philosopher's Stone",
    b2: "Chamber of Secrets",
    b3: "Prisoner of Azkaban",
    b4: "Goblet of Fire",
    b5: "Order of the Phoenix",
    b6: "Half-Blood Prince",
    b7: "Deathly Hallows"
}

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
            <div style={{height : '70vh', overflowY:'scroll'}}>
             <Tabs defaultActiveKey="info" >
                <Tab eventKey="info" title="Relation Info">

                    <Col>
                        {/* <Row>
                            <ButtonGroup>
                                <Button variant = 'outline-secondary' >source</Button>
                                <Button variant = 'outline-secondary' >{relation.top_sentence}</Button>
                                <Button variant = 'outline-secondary' >target</Button>
                            </ButtonGroup>
                        </Row> */}
                        <Row>
                        <ButtonGroup>
                                <Button variant = 'outline-secondary' >size</Button>
                                <Button variant = 'outline-secondary' >{relation.size}</Button>
                            </ButtonGroup>
                        </Row>
                        <Row>

                            {relation.key_words.map(kw => {
                                        return (<Button variant = 'outline-secondary' style={{margin : '2px'}}>{kw}</Button>)
                            }
                                
                                )}
                        </Row>
                    </Col>

                </Tab>
                <Tab eventKey="profile" title="Examples">
                    <Col>
                        {relation.relations.map( r => {
                            return (
                                <Row>
                                    <ButtonGroup className="d-flex">
                                        <Button variant = 'outline-secondary'>{r.s}</Button>


                                        <OverlayTrigger
                                            trigger="click"
                                            key='paragraph'
                                            placement='top'
                                            overlay={
                                                <Popover id={`popover-positioned-top`}>
                                                <Popover.Title as="h3">Book: {book_info[r.bi.book]} Chapter : {r.bi.chapter.slice(1)} Page :{r.bi.page.slice(1)}</Popover.Title>
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
                                </Row>
                            )
                    })}     
                    </Col>            
                </Tab>

                </Tabs>
                </div>

            
        )
    }


}

class AgeView extends React.Component {

    constructor(props) {
        super(props);
    }

    render () {

        var age = this.props.age;
        var age_freq = Object.entries(age.relation_frequency)

        return (


                <div style={{height : '70vh', overflowY:'scroll'}}>
                <Tabs >
                <Tab eventKey="info" title="Relation Info">

                    <Col>
                        <Row>


                        </Row>
                        <Row>
                        <ButtonGroup>
                                <Button variant = 'outline-secondary' >Frequency</Button>
                                <Button variant = 'outline-secondary' style={{marginRight : '3px'}} >{age.frequency}</Button>
                        </ButtonGroup>
                        <ButtonGroup>
                                <Button variant = 'outline-secondary' >n Charcters</Button>
                                <Button variant = 'outline-secondary' >{age.n_characters}</Button>
                        </ButtonGroup>
                        </Row>
                        <Row>
                            <ButtonGroup>
                            {age_freq.map(c => {
                                console.log(c)
                                var scale = (c[1]/age.frequency)*10
                                
                                try {
                                    var color = this.props.config[c[0]].color
                                } catch {
                                    var color = 'white'
                                }
                                        return (<Button variant = 'outline-secondary' 
                                                        style={{backgroundColor : color, width: `${scale}%`}}></Button>)
                            })}
                            </ButtonGroup>

                        </Row>

                        <Row>

                            {age.characters.map(c => {
                                        return (<Button variant = 'outline-secondary' style={{margin : '3px'}}>{c}</Button>)
                            }
                                
                                )}
                        </Row>
                    </Col>

                </Tab>
                <Tab eventKey="profile" title="Examples">
                    <Col>
                        {age.relations.map( r => {
                            return (
                                <Row>
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
                                                        textToHighlight={r.rc}
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
                                </Row>
                            )
                    })}     
                    </Col>            
                </Tab>
                </Tabs>
                </div>
            
        )
    }


}

class ConnView extends React.Component {

    constructor(props) {
        super(props);
    }

    render () {

        var conn = this.props.conn;
        var frequency = conn.relations.length
        var conn_freq = Object.entries(conn.relation_frequency)

        return (


                <div style={{height : '70vh', overflowY:'scroll'}}>
                <Tabs >
                <Tab eventKey="info" title="Relation Info">

                    <Col>
                        <Row>
                        <ButtonGroup>
                                <Button variant = 'outline-secondary' >Frequency</Button>
                                <Button variant = 'outline-secondary' style={{marginRight : '3px'}} >{frequency}</Button>
                        </ButtonGroup>
                        </Row>
                        <Row>
                            <ButtonGroup>
                            {conn_freq.map(c => {
                                // console.log(c)
                                var scale = (c[1]/conn.frequency)*10
                                
                                try {
                                    var color = this.props.config[c[0]].color
                                } catch {
                                    var color = 'white'
                                }
                                        return (<Button variant = 'outline-secondary' 
                                                        style={{backgroundColor : color, width: `${scale}%`}}></Button>)
                            })}
                            </ButtonGroup>

                        </Row>

                    </Col>

                </Tab>
                <Tab eventKey="profile" title="Examples">
                    <Col>
                        {conn.relations.map( r => {
                            return (
                                <Row>
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
                                                        textToHighlight={r.rc}
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
                                </Row>
                            )
                    })}     
                    </Col>            
                </Tab>
                </Tabs>
                </div>
            
        )
    }


}

class DataBuilder extends React.Component {

    constructor(props) {
        super(props);

        var config = JSON.parse(require('./config.json'));
        var data = JSON.parse(require('./data.json'));
        var relationslist = JSON.parse(require('./relations.json'));
        var ages = JSON.parse(require('./ages.json'));
        var conn = JSON.parse(require('./connections.json'));
        
        this.state = {
            config : this.buildconfig(config),
            data : this.builddata(data),
            query : [],
            newQuery : null,
            newData : true,
            relations : config.EdgeTypes,
            relationview : null,
            relationid : null,
            relationslist :relationslist,
            agesResource : ages,
            connResource : conn
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

    handleAgeQuery = (event) => {
        // this.state.query.push(event);
        // this.setState({ newQuery : true});
        // this.props.parentCallback({query: this.state.query});
        // event.preventDefault();
    }


    onSelectAge = (e) => {
        this.setState({age : e})   
        
    }

    onSelectConn = (e) => {
        this.setState({conn : e})   
        
    }

    // componentDidMount () {
    //     this.props.parentCallback({data: this.state.data,
    //         config: this.state.config});
    // }

    render () {

        var relationview = (this.state.relationid) ? <RelationView ref = {this.props.ref} relations = {this.state.relationslist[this.state.relationid]} /> : <div></div>
        var ageview = (this.state.age) ? <AgeView config = {this.state.config.EdgeTypes} age = {this.state.age} /> : <div></div>
        var connview = (this.state.conn) ? <ConnView config = {this.state.config.EdgeTypes} conn = {this.state.conn} /> : <div></div>
        return (

            <Container>
                <Tab.Container id="left-tabs-example" defaultActiveKey="first">
                    <Row>
                        <Nav variant="tabs" className="flex-row">
                            <Nav.Item>
                            <Nav.Link eventKey="first">Relations</Nav.Link>
                            </Nav.Item>
                            <Nav.Item>
                            <Nav.Link eventKey="second">Ages</Nav.Link>
                            </Nav.Item>
                            <Nav.Item>
                            <Nav.Link eventKey="third">Connections</Nav.Link>
                            </Nav.Item>
                        </Nav>
                    </Row>
                    <Row>
                    <Tab.Content style={{margin : '20px'}}>
                            <Tab.Pane eventKey="first">

                                <Row>
                                {this.state.relations.map(relation => {
                                    return (
                                            <Button 
                                                variant = 'outline-secondary' 
                                                style = {{backgroundColor : relation.color, width : '100px', color : "white", margin: '3px'}} 
                                                onClick = {(event) => this.onSelectRelation(relation)}>{relation.typeText}
                                            </Button>

                                            )
                                })}
                                </Row>
                                <Row>
                                <Col>{relationview}</Col>
                                </Row>
                            </Tab.Pane>
                            <Tab.Pane eventKey="second" >
                                <Row>
                                {this.state.agesResource.map(age => {
                                    return (
                                        
                                        <Button variant = 'outline-secondary' 
                                                style={{margin : '3px'}} 
                                                onClick = {(event) => this.onSelectAge(age)}>{age.id}</Button>
                                        
                                            )
                                })}
                                </Row>
                                <Row>
                                <Col>{ageview}</Col>
                                </Row>
                            </Tab.Pane>
                            <Tab.Pane eventKey="third" >
                                <Row style={{height : '30vh', overflowY:'scroll'}}>
                                {this.state.connResource.map(conn => {
                                    return (
                                        
                                        <Button variant = 'outline-secondary' 
                                                style={{margin : '3px'}} 
                                                onClick = {(event) => this.onSelectConn(conn)}>{conn.id}</Button>
                                        
                                            )
                                })}
                                </Row>
                                <Row>
                                    <Col>
                                {connview}
                                    </Col>
                                </Row>
                            </Tab.Pane>
                        </Tab.Content>
                    </Row>
                    </Tab.Container>
            </Container>
            
        )}
}


export default DataBuilder;