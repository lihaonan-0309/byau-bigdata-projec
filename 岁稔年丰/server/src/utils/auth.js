const jwt = require('jsonwebtoken')
const config = require('../config')

const secret = config.jwt.secret

function createToken(data) {
  const token = jwt.sign(
    {
      data
    },
    secret, {
    expiresIn: '30d' // 过期时间 15天
  })

  return token
}

module.exports = {
  createToken
}