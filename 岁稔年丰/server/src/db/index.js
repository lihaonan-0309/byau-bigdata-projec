//connect mongoose
const mongoose = require('mongoose')
const config = require('../config')
mongoose.Promise = global.Promise

let mongoUrl = "";

if (config.mongodb.user && config.mongodb.password) {
  mongoUrl = `mongodb://${config.mongodb.user}:${config.mongodb.password}@${config.mongodb.host}:${config.mongodb.port}/${config.mongodb.database}`
} else {
  // no account
  mongoUrl = `mongodb://${config.mongodb.host}:${config.mongodb.port}/${config.mongodb.database}`
}

console.log('mongodb connecting...')
mongoose.connect(mongoUrl)
const db = mongoose.connection

db.on('error', () => {
  console.log('connect mongodb error')
})

db.once('open', () => {
  console.log('connect mongodb successful')
})

module.exports = {
  db
}