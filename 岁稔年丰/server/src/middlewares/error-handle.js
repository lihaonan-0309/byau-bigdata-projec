
async function errorHandle(ctx, next) {
  try {
    await next()
  } catch (err) {
    console.error('server error:', err.message, ctx)
    // console.log(err.message);
    if (err.message == "Authentication Error") {
      ctx.result({
        code: 401,
        message: "invalid token"
      })
      return
    }

    ctx.err({
      message: "server error"
    })
    // ctx.app.emit('error', error, ctx)
  }
}

module.exports = {
  errorHandle
}