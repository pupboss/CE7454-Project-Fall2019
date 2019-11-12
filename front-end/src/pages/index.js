import { connect } from 'dva';
import { Select, Icon, message, Upload, Row, Col, Rate, Form, Radio, InputNumber, Button } from 'antd';
import { Chart, Geom, Axis, Tooltip, Legend } from 'bizcharts';
import styles from './index.css';

const FormItem = Form.Item
const { Option } = Select

const formItemLayout = {
  labelCol: {
    xs: { span: 24 },
    sm: { span: 6 },
  },
  wrapperCol: {
    xs: { span: 24 },
    sm: { span: 12 },
  },
}

const tailFormItemLayout = {
  wrapperCol: {
    xs: {
      offset: 24,
    },
    sm: {
      offset: 6,
    },
  },
}

const cols = {
  probability: { alias: 'Probability' },
  status: { alias: 'Status' }
};

const getBase64 = (img, callback) => {
  const reader = new FileReader();
  reader.addEventListener('load', () => callback(reader.result));
  reader.readAsDataURL(img);
}

const beforeUpload = (file) => {
  const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
  if (!isJpgOrPng) {
    message.error('You can only upload JPG/PNG file!');
  }
  const isLt2M = file.size / 1024 / 1024 < 2;
  if (!isLt2M) {
    message.error('Image must smaller than 2MB!');
  }
  return isJpgOrPng && isLt2M;
}

const genres = ['\N', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Music', 'Crime', 'Thriller', 'Adventure', 'Animation', 'Action', 'Biography', 'Horror', 'Mystery', 'Sci-Fi', 'Family', 'Sport', 'War', 'Documentary', 'History', 'Western', 'Musical', 'News']
const ones = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
const genreOptions = []
for (let i = 0; i < genres.length; i++) {
  genreOptions.push(<Option key={ones[i]}>{genres[i]}</Option>);
}

const directors = ['Christopher Nolan', 'Steven Spielberg', 'Jonathan Demme', 'Sidney Lumet', 'M. Night Shyamalan', 'Todd Phillips', 'Tony Kaye', 'Lana Wachowski', 'John Lasseter', 'George Lucas', 'Irvin Kershner', 'Billy Wilder', 'Jason Reitman', 'Spike Jonze', 'Baz Luhrmann', 'Gary Ross', 'Kathryn Bigelow']
const directorOptions = []
for (let i = 0; i < directors.length; i++) {
  directorOptions.push(<Option key={i.toString(9) + i}>{directors[i]}</Option>);
}

const actors = ['Leonardo DiCaprio', 'Talia Shire', 'Diane Keaton', 'Keira Knightley', 'Morgan Freeman', 'Brad Pitt', 'Al Pacino', 'Harrison Ford', 'Uma Thurman', 'Amy Adams', 'Marion Cotillard', 'Jessica Chastain', 'Johnny Depp', 'Bruce Willis', 'Ralph Fiennes', 'Jack Nicholson', 'Samuel L. Jackson', 'Robin Wright', 'Tilda Swinton']
const actorOptions = []
for (let i = 0; i < actors.length; i++) {
  actorOptions.push(<Option key={i.toString(11) + i}>{actors[i]}</Option>);
}

const Page = ({ dispatch, prediction, form }) => {
  const { getFieldDecorator } = form

  const uploadButton = (
    <div>
      <Icon type={prediction.poster.loading ? 'loading' : 'plus'} />
      <div className="ant-upload-text">Upload</div>
    </div>
  );

  const handleSubmit = (e) => {
    e.preventDefault()
    form.validateFieldsAndScroll((err, values) => {
      if (!err) {
        dispatch({
          type: 'prediction/predict',
          payload: {
            ...values,
          },
        })
      }
    })
  }

  const handleChange = (info) => {
    if (info.file.status === 'uploading') {
      dispatch({
        type: 'prediction/updateUserData',
        payload: {
          poster: {
            loading: true,
            imageUrl: null,
          },
        },
      })
      return;
    }
    if (info.file.status === 'done') {
      // Get this url from response in real world.
      getBase64(info.file.originFileObj, (imageUrl) => {
        dispatch({
          type: 'prediction/updateUserData',
          payload: {
            poster: {
              loading: false,
              imageUrl: imageUrl,
            },
          },
        })
      });
    }
  };

  return (
    <div className={styles.normal}>
      <div>
        <Form {...formItemLayout} onSubmit={handleSubmit}>
          <FormItem label="Category">
            {getFieldDecorator('category', {
              rules: [{ type: 'integer', required: true, message: 'Please select an action', whitespace: true }],
              initialValue: 0,
            })(
              <Radio.Group disabled>
                <Radio value={0}>Movie</Radio>
                <Radio value={1}>Drama</Radio>
                <Radio value={2}>Other</Radio>
              </Radio.Group>
            )}
          </FormItem>
          <FormItem label="Poster">
            <Upload
              name="avatar"
              listType="picture"
              className="avatar-uploader"
              showUploadList={false}
              action="https://www.mocky.io/v2/5cc8019d300000980a055e76"
              beforeUpload={beforeUpload}
              onChange={handleChange}
            >
              {prediction.poster.imageUrl ? <img src={prediction.poster.imageUrl} alt="avatar" style={{ width: '100%' }} /> : uploadButton}
            </Upload>
          </FormItem>
          <FormItem label="Year">
            {getFieldDecorator('year', {
              rules: [{ required: true, message: 'Please input the year' }],
              initialValue: 2019,
            })(
              <InputNumber />
            )}
          </FormItem>
          <FormItem label="Duration">
            {getFieldDecorator('duration', {
              rules: [{ required: true, message: 'Please input the duration' }],
              initialValue: 120,
            })(
              <InputNumber />
            )}
            <span className="ant-form-text"> mins</span>
          </FormItem>
          <FormItem label="Director">
            {getFieldDecorator('director', {
              rules: [{ required: true, message: 'Please select the director' }],
            })(
                <Select
                  showSearch
                  style={{ width: 200 }}
                  placeholder="Select a director"
                >
                  {directorOptions}
                </Select>
            )}
          </FormItem>
          <FormItem label="Actors">
            {getFieldDecorator('actors', {
              rules: [{ required: true, message: 'Please select some actors' }],
            })(
              <Select
                mode="multiple"
                style={{ width: '70%' }}
                placeholder="Please select"
              >
                {actorOptions}
              </Select>
            )}
          </FormItem>
          <FormItem label="Budget">
            {getFieldDecorator('budget', {
              rules: [{ required: true, message: 'Please input your budget' }],
              initialValue: 2000000,
            })(
              <InputNumber step={100000} style={{ width: '140px' }} />
            )}
            <span className="ant-form-text"> USD</span>
          </FormItem>
          <FormItem label="Genres">
            {getFieldDecorator('genres', {
              rules: [{ required: true, message: 'Please select some genres' }],
            })(
              <Select
                mode="multiple"
                style={{ width: '70%' }}
                placeholder="Please select"
              >
                {genreOptions}
              </Select>
            )}
          </FormItem>
          <FormItem {...tailFormItemLayout}>
            <Button type="primary" htmlType="submit">
              Submit
            </Button>
          </FormItem>
        </Form>
      </div>
      <div style={{ paddingTop: '50px' }}>
        <Row>
          <Col span={12}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ paddingTop: '90px' }}>
                Predict: <Rate allowHalf disabled count={10} value={parseInt(prediction.rating / 0.5) * 0.5} /> {prediction.rating.toFixed(2)} Stars
              </div>
            </div>
          </Col>
          <Col span={12}>
            <div>
              <Chart width={600} height={400} data={prediction.boxOfficeData} scale={cols}>
                <Axis name="status" title/>
                <Axis name="probability" title/>
                <Legend position="top" dy={-20} />
                <Tooltip />
                <Geom type="interval" position="status*probability" color="status" />
              </Chart>
            </div>
          </Col>
        </Row>
      </div>
    </div>
  );
}

export default connect(({ prediction }) => ({ prediction }))(Form.create()(Page))
