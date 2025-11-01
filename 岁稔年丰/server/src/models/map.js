const mongoose = require('mongoose')
const Schema = mongoose.Schema
mongoose.Model

const mapSchema = new Schema({
    date: String,
    link: String,
    list: Array,
})

const mapModel = mongoose.model('mapdatas', mapSchema)

module.exports = {
    mapModel
}