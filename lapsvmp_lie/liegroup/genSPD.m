% calculate a SPD matrix of given image
function c = genSPD(img,varargin)

ip=inputParser;
addParameter(ip,'dim',8);
addParameter(ip,'imgtype','gray')

parse(ip,varargin{:});
dim=ip.Results.dim;
imgtype=ip.Results.imgtype;


if strcmp(imgtype,'gray')

    if(length(size(img))>2)
    img = rgb2gray(img);
    end

    img = double(img);
    
    bb = 1;
    if bb==1
        [ii,jj]=find(img~=0);
        imin = min(ii);
        imax = max(ii);
        jmin = min(jj);
        jmax = max(jj);
        img=img(imin:imax,jmin:jmax);
    end

    h1 = fspecial('average',[3 3]);
    h2 = fspecial('disk',1);
    h3 = fspecial('gaussian',[3 3]);
    h4 = fspecial('laplacian');
    h5 = fspecial('log',[3 3]);
    h6 = [-1 -1 -1;-1 9 -1;-1 -1 -1];
    h7 = fspecial('prewitt');
    h8 = fspecial('sobel');
    h9 = [-1 0 1;-2 0 2;-1 0 1];
    
    g1=imfilter(img,h1,'same');
    g2=imfilter(img,h2,'same');
    g3=imfilter(img,h3,'same');
    g4=imfilter(img,h4,'same');
    g5=imfilter(img,h5,'same');
    g6=imfilter(img,h6,'same');
    g7=imfilter(img,h7,'same');
    g8=imfilter(img,h8,'same');
    g9=imfilter(img,h9,'same');

    [Fx, Fy] = gradient(img);
    [Fxx, Fxy] = gradient(Fx);
    [Fyx, Fyy] = gradient(Fy);
    
    aFx = abs(Fx);
    aFy = abs(Fy);
    aFxx = abs(Fxx);
    aFxy = abs(Fxy);
    aFyx = abs(Fyx);
    aFyy = abs(Fyy);



    % GABOR
    wavelength = 4;
    orientation = 90;
    [mag,phase] = imgaborfilt(img,wavelength,orientation);
%     phase=phase./255;
%     g3=g3./255;
%     g4=g4./255;
%     g8=g8./255;
%     g9=g9./255;
%     img=img./255;

    %compute mean
    [row, col] = size(img);
    m=zeros(dim);
    va=zeros(1,dim);
    for y=1:row
        for x=1:col
            if(dim == 9)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x), sqrt(aFx(y,x)*aFx(y,x)+aFy(y,x)*aFy(y,x)), aFxx(y,x), aFyy(y,x), atan(max(aFx(y,x),0.00000001)/max(aFy(y,x),0.00000001))];
            elseif (dim == 8)
                v=[y./row,x./col, img(y,x), g8(y,x), g9(y,x), sqrt(g8(y,x)*g8(y,x)+g9(y,x)*g9(y,x)), g4(y,x),phase(y,x)];
            elseif (dim == 7)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x), aFxx(y,x), aFyy(y,x)];
            elseif (dim == 6)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x), sqrt(aFx(y,x)*aFx(y,x)+aFy(y,x)*aFy(y,x))];
            elseif (dim == 5)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x)];
            else
                v=[g3(y,x),g8(y,x), img(y,x)];
            end
            va=va+v;
%             m=m+v'*v;
        end
    end
    meanva=va/(row*col);

    %compute SPD
    for y=1:row
        for x=1:col
            if(dim == 9)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x), sqrt(aFx(y,x)*aFx(y,x)+aFy(y,x)*aFy(y,x)), aFxx(y,x), aFyy(y,x), atan(max(aFx(y,x),0.00000001)/max(aFy(y,x),0.00000001))];
            elseif (dim == 8)
                v=[y./row,x./col, img(y,x), g8(y,x), g9(y,x), sqrt(g8(y,x)*g8(y,x)+g9(y,x)*g9(y,x)), g4(y,x),phase(y,x)];
            elseif (dim == 7)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x), aFxx(y,x), aFyy(y,x)];
            elseif (dim == 6)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x), sqrt(aFx(y,x)*aFx(y,x)+aFy(y,x)*aFy(y,x))];
            elseif (dim == 5)
                v=[g3(y,x),g8(y,x), img(y,x), aFx(y,x), aFy(y,x)];
            else
                v=[g3(y,x),g8(y,x), img(y,x)];
            end
            m=m+(v-meanva)'*(v-meanva);
        end
    end

    c = m/(row*col-1);
elseif strcmp(imgtype,'rgb')

    img = double(img);

    r=img(:,:,1);
    g=img(:,:,2);
    b=img(:,:,3);

    [Fx, Fy] = gradient(img);
    aFx=abs(mean(Fx,3));
    aFy=abs(mean(Fy,3));

    [row, col] = size(r);
    m=zeros(dim);
    for y=1:row
        for x=1:col
            v=[y, x, r(y,x), g(y,x), b(y,x), aFx(y,x), aFy(y,x), sqrt(aFx(y,x)*aFx(y,x)+aFy(y,x)*aFy(y,x))];
            m=m+v'*v;
        end
    end

    c = m/(row*col);
end