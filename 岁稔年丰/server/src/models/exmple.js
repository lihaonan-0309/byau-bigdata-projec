const mongoose = require('mongoose')
const Schema = mongoose.Schema
mongoose.Model

const exampleSchema = new Schema({
  cid: Number,
  name: String,
  list: Array,
})

const exampleModel = mongoose.model('chartData', exampleSchema)

module.exports = {
  exampleModel
}