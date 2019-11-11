import qs from 'qs'
import request from '../utils/request'

const ROOT = 'https://api.apiopen.top'

export async function show() {
  const response = await request(`${ROOT}/recommendPoetry`, {
    method: 'GET',
  })
  return response
}

export async function registerFogNode(publicKey, secret) {
  const params = qs.stringify({ publicKey, secret }, { skipNulls: true })
  const response = await request(`${ROOT}/fognode/register`, {
    method: 'POST',
    body: params,
  })
  return response
}

export async function unlinkFogNode(publicKey) {
  const params = qs.stringify({ publicKey }, { skipNulls: true })
  const response = await request(`${ROOT}/fognode/unlink`, {
    method: 'POST',
    body: params,
  })
  return response
}

export async function getFogNodeData() {
  const response = await request(`${ROOT}/fognode/list`, {
    method: 'GET',
  })
  return response
}
