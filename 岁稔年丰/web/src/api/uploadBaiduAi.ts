import { request } from '../request/index'
interface Params {
    typeId: Number
}

export function uploadBaiduAi(file: FormData, p: Params) {
    return request({
        method: 'post',
        url: '/upload/image/baidu',
        data: file,
        params: p
    })
}
