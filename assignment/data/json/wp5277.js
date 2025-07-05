const pathways = require('./wp5277.json')

const entities = Object.entries(pathways['entitiesById'])
  .filter(([key, _]) => !(key.startsWith('http') || key.startsWith('publication')))
  .map(([key, value]) => {
    return value
  })

const dataNodes = entities.filter(({gpmlElementName}) => gpmlElementName === 'DataNode')
  .map(({id, }))
const edges = entities.filter(({gpmlElementName}) => gpmlElementName === 'Interaction')

console.log(dataNodes)