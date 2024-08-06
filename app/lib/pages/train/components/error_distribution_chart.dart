import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:app/api/api.dart';

class ErrorDistributionChart extends StatefulWidget {
  final String sessionName;

  const ErrorDistributionChart({super.key, required this.sessionName});

  @override
  State createState() => _ErrorDistributionChartState();
}

class _ErrorDistributionChartState extends State<ErrorDistributionChart> {
  List<BarChartGroupData> _chartData = [];

  @override
  void initState() {
    super.initState();
    _fetchData();
  }

  Future<void> _fetchData() async {
    try {
      final response = await API.getErrorDistribution(widget.sessionName);
      setState(() {
        _chartData = _createChartData(response['distribution']);
      });
    } catch (e) {
      debugPrint('Error fetching error distribution: $e');
    }
  }

  List<BarChartGroupData> _createChartData(List<dynamic> distribution) {
    return List.generate(distribution.length, (index) {
      final item = distribution[index];
      return BarChartGroupData(
        x: index,
        barRods: [
          BarChartRodData(
            toY: item['count'].toDouble(),
            color: Colors.blue,
          ),
        ],
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('Error Distribution for Session: ${widget.sessionName}',
            style: Theme.of(context).textTheme.headlineMedium),
        const SizedBox(height: 20),
        SizedBox(
          height: 300,
          child: BarChart(
            BarChartData(
              alignment: BarChartAlignment.spaceAround,
              maxY: _chartData.isEmpty ? 10 : null,
              barGroups: _chartData,
              titlesData: FlTitlesData(
                show: true,
                bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                  showTitles: true,
                  getTitlesWidget: (value, meta) {
                    int index = value.toInt();
                    if (index % 2 == 0 && index < _chartData.length) {
                      return Text('${index * 0.5}-${(index + 1) * 0.5}');
                    }
                    return const Text('');
                  },
                )),
                leftTitles:
                    const AxisTitles(sideTitles: SideTitles(showTitles: true)),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
