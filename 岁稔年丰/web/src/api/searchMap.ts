import { request } from '../request/index'

export function searchMap() {
    return request({
        method: 'get',
        url: '/map',
    })
}
