#ifndef PLOTS_INCLUDED
#define PLOTS_INCLUDED

#include "GeometryVector.h"
#include <vector>

//functions and classes to output xmGrace/qtGrace figures
//Use the class if you want even more control
extern std::string PlotFunction_Grace_SetType;
void PlotFunction_Grace(const std::vector<GeometryVector> * presult, size_t NumSet, const std::string & OutFilePrefix, const std::string & xLabel, const std::string & yLabel, const std::vector<std::string> & legends, const std::string & Title, double MinX, double MaxX, double TickX, double MinY, double MaxY, double TickY);
void PlotFunction_Grace(const std::vector<GeometryVector> * presult, size_t NumSet, const std::string & OutFilePrefix, const std::string & xLabel, const std::string & yLabel, const std::vector<std::string> & legends, const std::string & Title);
void PlotFunction_Grace(const std::vector<GeometryVector> & result, const std::string & OutFilePrefix="temp", const std::string & xLabel="", const std::string & yLabel="", const std::string & Title="");


class gracePlot
{
	const int NColor = 14;
	const int NLineStyle = 8;
	const int NSymbol = 10;
public:
	class dataSet
	{
	public:
		std::vector<GeometryVector> result;
		std::string type;//xy, xydy, or xydxdy
		int lineColor;
		int lineStyle;
		int symbolColor;
		int symbolStyle;
		std::string legend;
		double symbolSize;
	};
	std::vector<dataSet> dataSets;
	double MinX, MaxX, TickX, MinY, MaxY, TickY;
	std::string xLabel, yLabel, Title;
	std::string xScale, yScale;//"Normal" or "Logarithmic"
	gracePlot() : MinX(0.0), MaxX(1.0), TickX(0.5), MinY(0.0), MaxY(1.0), TickY(0.5), xScale("Normal"), yScale("Normal")
	{}
	void autoScale();
	void autoTick();

	void autoScaleAndTick();
	void addDataSet(std::vector<GeometryVector> data, std::string legend = "", int lineColor = -1, int lineStyle = -1, int symbolColor = -1, int symbolStyle = -1, std::string type = "xy", double symbolSize = 1.0); // for colors and styles, -1 means auto, and 0 means none
	void outputFigure(const std::string & OutFilePrefix);
};


//ToDo : revamp them as constructors of class GracePlot
void ReadGraceData(std::vector<std::vector<GeometryVector> > & result, const std::string & InFilePrefix);
void ReadGraceData(std::vector<std::vector<GeometryVector> > & result, std::istream & ifile);

#endif