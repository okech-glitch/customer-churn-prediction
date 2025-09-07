import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { apiService } from '../services/api';

const ModelPerformance = ({ onShowSnackbar }) => {
  const [modelInfo, setModelInfo] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelData();
  }, []);

  const fetchModelData = async () => {
    setLoading(true);
    try {
      const [modelsData, featuresData] = await Promise.all([
        apiService.getModels(),
        apiService.getFeatureImportance(),
      ]);
      
      setModelInfo(modelsData);
      setFeatureImportance(featuresData);
    } catch (err) {
      setError(err.message);
      onShowSnackbar(`Failed to load model data: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const getPerformanceColor = (score) => {
    if (score >= 0.9) return 'success';
    if (score >= 0.8) return 'warning';
    return 'error';
  };

  const getPerformanceLabel = (score) => {
    if (score >= 0.9) return 'Excellent';
    if (score >= 0.8) return 'Good';
    if (score >= 0.7) return 'Fair';
    return 'Poor';
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  // Prepare data for charts
  const modelPerformanceData = modelInfo ? Object.entries(modelInfo).map(([name, info]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    auc: info.auc_score || 0,
    accuracy: info.accuracy || 0,
  })) : [];

  const featureImportanceData = featureImportance?.feature_importance ? 
    Object.entries(featureImportance.feature_importance)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([name, importance]) => ({
        name: name.replace(/([A-Z])/g, ' $1').trim(),
        importance: importance,
      })) : [];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <Grid container spacing={3}>
      {/* Model Performance Overview */}
      <Grid item xs={12}>
        <Card>
          <CardHeader title="Model Performance Overview" />
          <CardContent>
            {modelInfo ? (
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={modelPerformanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip formatter={(value) => [value.toFixed(3), 'Score']} />
                      <Legend />
                      <Bar dataKey="auc" fill="#8884d8" name="AUC Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </Grid>
                <Grid item xs={12} md={4}>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell align="right">AUC</TableCell>
                          <TableCell align="right">Status</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(modelInfo).map(([name, info]) => (
                          <TableRow key={name}>
                            <TableCell component="th" scope="row">
                              {name.charAt(0).toUpperCase() + name.slice(1)}
                            </TableCell>
                            <TableCell align="right">
                              {(info.auc_score || 0).toFixed(3)}
                            </TableCell>
                            <TableCell align="right">
                              <Chip
                                label={getPerformanceLabel(info.auc_score || 0)}
                                color={getPerformanceColor(info.auc_score || 0)}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            ) : (
              <Alert severity="info">
                No model performance data available.
              </Alert>
            )}
          </CardContent>
        </Card>
      </Grid>

      {/* Feature Importance */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Feature Importance" />
          <CardContent>
            {featureImportanceData.length > 0 ? (
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Model: {featureImportance?.model_name || 'Unknown'}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {featureImportanceData.map((feature, index) => (
                    <Box key={feature.name} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">{feature.name}</Typography>
                        <Typography variant="body2">
                          {(feature.importance * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={feature.importance * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'grey.200',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: COLORS[index % COLORS.length],
                          },
                        }}
                      />
                    </Box>
                  ))}
                </Box>
              </Box>
            ) : (
              <Alert severity="info">
                No feature importance data available.
              </Alert>
            )}
          </CardContent>
        </Card>
      </Grid>

      {/* Model Details */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Model Details" />
          <CardContent>
            {modelInfo ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Available Models
                </Typography>
                {Object.entries(modelInfo).map(([name, info]) => (
                  <Box key={name} sx={{ mb: 2, p: 2, border: 1, borderColor: 'grey.300', borderRadius: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      {name.charAt(0).toUpperCase() + name.slice(1)}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                      <Chip
                        label={`AUC: ${(info.auc_score || 0).toFixed(3)}`}
                        color={getPerformanceColor(info.auc_score || 0)}
                        size="small"
                      />
                      {info.accuracy && (
                        <Chip
                          label={`Accuracy: ${(info.accuracy * 100).toFixed(1)}%`}
                          variant="outlined"
                          size="small"
                        />
                      )}
                    </Box>
                    {info.features && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        Features: {info.features.length}
                      </Typography>
                    )}
                  </Box>
                ))}
              </Box>
            ) : (
              <Alert severity="info">
                No model details available.
              </Alert>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ModelPerformance;
