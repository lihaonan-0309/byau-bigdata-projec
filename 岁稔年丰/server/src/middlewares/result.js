async function result(ctx, next) {
  // response result fun
  ctx.result = ({ code, data, message }) => {
    ctx.status = 200
    ctx.body = {
      code,
      data,
      message   // 多用于错误信息
    }
  }
  // success result fun
  ctx.succ = ({ data, message }) => {
    ctx.result({
      code: 200,
      data,
      message
    })
  }
  // error result fun
  ctx.err = ({ data, message }) => {
    ctx.result({
      code: 500,
      data,
      message   // 错误信息
    })
  }

  await next()
}

module.exports = {
  result
}