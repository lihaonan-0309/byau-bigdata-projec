async function notFound(ctx, next) {
  await next()
  ctx.result({
    code: 404,
    message: "404 not found"
  })
}

module.exports = {
  notFound
}