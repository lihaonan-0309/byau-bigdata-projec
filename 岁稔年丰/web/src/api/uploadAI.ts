import { request } from '../request/index'

export function uploadAI(data: FormData, params: object) {
    return request({
        method: 'post',
        url: '/upload/image',
        data,
        params
    })
}
