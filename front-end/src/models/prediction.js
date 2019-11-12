import * as expServices from '../services/prediction'

const initialState = {
  poster: {
    loading: false,
    imageUrl: null,
  },
  boxOfficeData: [
    { status: 'Loss', probability: 0.15 },
    { status: 'Flat', probability: 0.8 },
    { status: 'Profit', probability: 0.05 }
  ],
}

export default {
  namespace: 'prediction',

  state: initialState,

  effects: {
    * show(action, { call, put }) {
      const { data } = yield call(expServices.show)
      console.log(data)
      if (data) {
        yield put({
          type: 'updateUserData',
          payload: {
            ...data,
          },
        })
      } else {
        yield put({ type: 'showFail' })
      }
    },
  },

  reducers: {
    updateUserData: (state, { payload }) => ({
      ...state,
      ...payload,
    }),
  },
}
