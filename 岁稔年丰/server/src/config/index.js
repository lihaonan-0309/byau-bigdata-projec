const constant = require('../constant')
const mongodb = {
  user: "admin",
  password: "lilili999",
  host: "43.138.11.133",
  port: 26666,
  database: "myAgriculture"
}

const jwt = {
  secret: "secret886886",
  unless: [
    // /^\/api\/users\/login/
    "/",
    `/api/${constant.API_VERSION}/example`,
    `/api/${constant.API_VERSION}/example/`,
    `/api/${constant.API_VERSION}/upload/image`
  ]
}

module.exports = {
  mongodb,
  jwt
}