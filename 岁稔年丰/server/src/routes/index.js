const Router = require('koa-router')   // router
const fs = require('fs')
const constant = require('../constant')

const API_VERSION = constant.API_VERSION

function autoImportRoutes(parentRoute) {
  // auto import routes to apiRootRoute
  const files = fs.readdirSync(__dirname)
  const jsFiles = files.filter(f => f.endsWith('.js'))
  //批量导入
  jsFiles.forEach(f => {
    const mapping = require(`${__dirname}/${f}`)
    // 过滤出router对象
    for (let key in mapping) {
      const obj = mapping[key]
      if (obj.constructor == Router) {
        // 导入至apiRootRoute
        console.log(`auto import route: ${f}`);
        const route = obj
        parentRoute.use(route.routes()).use(route.allowedMethods())
      }
    }
  })
}

// api root route
const apiRootRoute = new Router({
  prefix: `/api/${API_VERSION}`
})
// auto import routes to apiRootRoute
autoImportRoutes(apiRootRoute)

// base route
const baseRoute = new Router()
baseRoute.get('/', async (ctx, next) => {
  await next()
  ctx.succ({ message: "Welcome to the API" })
})
baseRoute.use(apiRootRoute.routes()).use(apiRootRoute.allowedMethods())
// other route
// ...
// baseRoute.use(otherRoute.routes()).use(otherRoute.allowedMethods())

// export default baseRoute
module.exports = {
  baseRoute
}