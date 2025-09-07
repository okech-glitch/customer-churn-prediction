import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Chip,
  CircularProgress,
  Alert,
  Divider,
} from '@mui/material';
import {
  Send as SendIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { apiService } from '../services/api';

const PredictionForm = ({ onShowSnackbar }) => {
  const [formData, setFormData] = useState({
    CreditScore: '',
    Geography: '',
    Gender: '',
    Age: '',
    Tenure: '',
    Balance: '',
    NumOfProducts: '',
    HasCrCard: '',
    IsActiveMember: '',
    EstimatedSalary: '',
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (field) => (event) => {
    setFormData({
      ...formData,
      [field]: event.target.value,
    });
    setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Convert form data to appropriate types
      const customerData = {
        CreditScore: parseInt(formData.CreditScore),
        Geography: formData.Geography,
        Gender: formData.Gender,
        Age: parseInt(formData.Age),
        Tenure: parseInt(formData.Tenure),
        Balance: parseFloat(formData.Balance),
        NumOfProducts: parseInt(formData.NumOfProducts),
        HasCrCard: parseInt(formData.HasCrCard),
        IsActiveMember: parseInt(formData.IsActiveMember),
        EstimatedSalary: parseFloat(formData.EstimatedSalary),
      };

      const result = await apiService.predictSingle(customerData);
      setPrediction(result);
      onShowSnackbar('Prediction completed successfully!', 'success');
    } catch (err) {
      setError(err.message);
      onShowSnackbar(`Prediction failed: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      CreditScore: '',
      Geography: '',
      Gender: '',
      Age: '',
      Tenure: '',
      Balance: '',
      NumOfProducts: '',
      HasCrCard: '',
      IsActiveMember: '',
      EstimatedSalary: '',
    });
    setPrediction(null);
    setError(null);
  };

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'High':
        return 'success';
      case 'Medium':
        return 'warning';
      case 'Low':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Grid container spacing={3}>
      {/* Input Form */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Customer Information" />
          <CardContent>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Credit Score"
                    type="number"
                    value={formData.CreditScore}
                    onChange={handleInputChange('CreditScore')}
                    required
                    inputProps={{ min: 300, max: 850 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Geography</InputLabel>
                    <Select
                      value={formData.Geography}
                      onChange={handleInputChange('Geography')}
                      label="Geography"
                    >
                      <MenuItem value="France">France</MenuItem>
                      <MenuItem value="Germany">Germany</MenuItem>
                      <MenuItem value="Spain">Spain</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Gender</InputLabel>
                    <Select
                      value={formData.Gender}
                      onChange={handleInputChange('Gender')}
                      label="Gender"
                    >
                      <MenuItem value="Male">Male</MenuItem>
                      <MenuItem value="Female">Female</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Age"
                    type="number"
                    value={formData.Age}
                    onChange={handleInputChange('Age')}
                    required
                    inputProps={{ min: 18, max: 100 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Tenure (years)"
                    type="number"
                    value={formData.Tenure}
                    onChange={handleInputChange('Tenure')}
                    required
                    inputProps={{ min: 0, max: 10 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Balance"
                    type="number"
                    value={formData.Balance}
                    onChange={handleInputChange('Balance')}
                    required
                    inputProps={{ min: 0, step: 0.01 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Number of Products"
                    type="number"
                    value={formData.NumOfProducts}
                    onChange={handleInputChange('NumOfProducts')}
                    required
                    inputProps={{ min: 1, max: 4 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Has Credit Card</InputLabel>
                    <Select
                      value={formData.HasCrCard}
                      onChange={handleInputChange('HasCrCard')}
                      label="Has Credit Card"
                    >
                      <MenuItem value={1}>Yes</MenuItem>
                      <MenuItem value={0}>No</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Is Active Member</InputLabel>
                    <Select
                      value={formData.IsActiveMember}
                      onChange={handleInputChange('IsActiveMember')}
                      label="Is Active Member"
                    >
                      <MenuItem value={1}>Yes</MenuItem>
                      <MenuItem value={0}>No</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Estimated Salary"
                    type="number"
                    value={formData.EstimatedSalary}
                    onChange={handleInputChange('EstimatedSalary')}
                    required
                    inputProps={{ min: 0, step: 0.01 }}
                  />
                </Grid>
              </Grid>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? 'Predicting...' : 'Predict Churn'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={handleReset}
                  disabled={loading}
                >
                  Reset
                </Button>
              </Box>
            </form>
          </CardContent>
        </Card>
      </Grid>

      {/* Prediction Results */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Prediction Results" />
          <CardContent>
            {prediction ? (
              <Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Churn Prediction
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Chip
                      label={prediction.prediction === 1 ? 'Will Churn' : 'Will Stay'}
                      color={prediction.prediction === 1 ? 'error' : 'success'}
                      size="large"
                    />
                    <Chip
                      label={`${(prediction.churn_probability * 100).toFixed(1)}%`}
                      color={getConfidenceColor(prediction.confidence)}
                      variant="outlined"
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {prediction.confidence}
                  </Typography>
                </Box>

                <Divider sx={{ my: 2 }} />

                <Box>
                  <Typography variant="h6" gutterBottom>
                    Probability Breakdown
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Stay Probability</Typography>
                      <Typography variant="body2">
                        {((1 - prediction.churn_probability) * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Churn Probability</Typography>
                      <Typography variant="body2">
                        {(prediction.churn_probability * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>

                  {/* Progress bar visualization */}
                  <Box sx={{ mt: 2 }}>
                    <Box
                      sx={{
                        height: 20,
                        backgroundColor: 'grey.200',
                        borderRadius: 1,
                        overflow: 'hidden',
                        display: 'flex',
                      }}
                    >
                      <Box
                        sx={{
                          width: `${(1 - prediction.churn_probability) * 100}%`,
                          backgroundColor: 'success.main',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <Typography variant="caption" color="white">
                          Stay
                        </Typography>
                      </Box>
                      <Box
                        sx={{
                          width: `${prediction.churn_probability * 100}%`,
                          backgroundColor: 'error.main',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <Typography variant="caption" color="white">
                          Churn
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                </Box>
              </Box>
            ) : (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: 200,
                  textAlign: 'center',
                }}
              >
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No Prediction Yet
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Fill in the customer information and click "Predict Churn" to get started.
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default PredictionForm;
