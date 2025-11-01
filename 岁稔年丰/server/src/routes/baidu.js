const Router = require('koa-router')
const { koaBody } = require('koa-body')
const { UploadController } = require('../controllers/upload.baidu')

const uploadRoute = new Router({
    prefix: '/upload'
})

uploadRoute.post('/image/baidu',
    koaBody({
        multipart: true,
        formidable: {
            maxFileSize: 200 * 1024 * 1024    // 设置上传文件大小最大限制，默认2M
        }
    }),
    UploadController.uploadImage)

module.exports = {
    uploadRoute
}