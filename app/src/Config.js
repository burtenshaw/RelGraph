
var config = JSON.parse(require('./config.json'));

// const PotterConfig = config.map();

console.log(config)

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
            <symbol viewBox="0 0 50 50" id={type.shapeId} key="0">
                <circle cx="25" cy="25" r="8" fill={type.color} > </circle>
            </symbol>
    )
    EdgeTypes[type.typeText] = type
})

const PotterConfig =  {
    NodeTypes: NodeTypes,
    NodeSubtypes: {},
    EdgeTypes: EdgeTypes
  }

export default PotterConfig;
