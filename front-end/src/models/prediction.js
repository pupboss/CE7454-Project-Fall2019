import * as predictionServices from '../services/prediction'

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
  rating: 8,
}

export default {
  namespace: 'prediction',

  state: initialState,

  effects: {
    * predict({ payload: { budget, year, duration, genres } }, { call, put }) {
      let genreString = ''
      for (let i = 0; i < genres.length; i++) {
        genreString += genres[i].toString() + ','
      }
      genreString = genreString.slice(0,-1)
      const { data } = yield call(predictionServices.predict, budget, year, duration, genreString)
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
