import qs from 'qs'
import request from '../utils/request'

const ROOT = 'http://localhost:5000'

export async function predict(budget, year, duration, genres) {
  const params = qs.stringify({ budget, year, duration, genres }, { skipNulls: true })
  const response = await request(`${ROOT}/predict`, {
    method: 'POST',
    body: params,
  })
  return response
}
